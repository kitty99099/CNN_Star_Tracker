#!/usr/bin/env python3
"""Run CNN star tracking from an MT9V022 through Arducam USB2 Shield.

Data path:

    MT9V022 -> Arducam USB Camera Shield -> USB host -> PYNQ Python -> DPU

This file intentionally uses the proprietary ``ArducamSDK`` API rather than
OpenCV ``VideoCapture``.  The USB Camera Shield is not assumed to be a UVC
camera.  The official MT9V022 640x480 MONO8 ``.cfg`` file is passed through
``--camera-config``.
"""

from __future__ import annotations

import argparse
import importlib
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple

import numpy as np


ARDUCAM_ERRORS = {
    0xFF05: "USB_CAMERA_BUFFER_ERROR",
    0xFF06: "USB_CAMERA_NOT_FOUND_DEVICE_ERROR",
    0xFF0B: "USB_CAMERA_I2C_BIT_ERROR",
    0xFF0C: "USB_CAMERA_I2C_NACK_ERROR",
    0xFF0D: "USB_CAMERA_I2C_TIMEOUT",
    0xFF20: "USB_CAMERA_USB_TASK_ERROR",
    0xFF21: "USB_CAMERA_DATA_OVERFLOW_ERROR",
    0xFF22: "USB_CAMERA_DATA_LACK_ERROR",
    0xFF23: "USB_CAMERA_FIFO_FULL_ERROR",
    0xFF24: "USB_CAMERA_DATA_LEN_ERROR",
    0xFF26: "USB_CAMERA_USB_TIMEOUT_ERROR",
    0xFF30: "USB_CAMERA_READ_EMPTY_ERROR",
}


def arducam_error(code: int) -> str:
    return f"{ARDUCAM_ERRORS.get(code, 'Arducam SDK error')} (0x{code:04X})"


def import_arducam_modules() -> Tuple[Any, Any]:
    try:
        sdk = importlib.import_module("ArducamSDK")
        parser = importlib.import_module("arducam_config_parser")
    except ImportError as exc:
        raise RuntimeError(
            "Arducam USB Camera Shield support requires both ArducamSDK and "
            "arducam_config_parser. Install versions compatible with the "
            "board CPU architecture and Python version."
        ) from exc
    return sdk, parser


class ArducamUsbCamera:
    """Minimal MT9V022 MONO8 capture wrapper around the official SDK."""

    def __init__(self, config_file: str, camera_index: int = 0):
        self.sdk, self.config_parser = import_arducam_modules()
        self.handle = None
        self.camera_cfg = None
        self.color_mode = 0
        self.running = False
        self.capture_error: Optional[RuntimeError] = None
        self.signal = threading.Condition()
        self.capture_thread: Optional[threading.Thread] = None
        self._open(config_file, camera_index)

    def _open(self, config_file: str, camera_index: int) -> None:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Arducam camera config not found: {config_path}")

        config = self.config_parser.LoadConfigFile(str(config_path))
        parameter = config.camera_param.getdict()
        width = int(parameter["WIDTH"])
        height = int(parameter["HEIGHT"])
        bit_width = int(parameter["BIT_WIDTH"])
        pixel_bytes = 2 if 8 < bit_width <= 16 else 1
        format_mode = parameter["FORMAT"][0]
        self.color_mode = parameter["FORMAT"][1]

        requested_cfg = {
            "u32CameraType": 0x00,
            "u32Width": width,
            "u32Height": height,
            "usbType": 0,
            "u8PixelBytes": pixel_bytes,
            "u16Vid": 0,
            "u32Size": 0,
            "u8PixelBits": bit_width,
            "u32I2cAddr": parameter["I2C_ADDR"],
            "emI2cMode": parameter["I2C_MODE"],
            "emImageFmtMode": format_mode,
            "u32TransLvl": parameter.get("TRANS_LVL", 0),
        }

        ret, handle, returned_cfg = self.sdk.Py_ArduCam_open(
            requested_cfg, camera_index
        )
        if ret != 0:
            raise RuntimeError(f"Failed to open Arducam: {arducam_error(ret)}")

        self.handle = handle
        self.camera_cfg = returned_cfg
        try:
            self._apply_configuration(config)
            if returned_cfg["emImageFmtMode"] != self.sdk.FORMAT_MODE_MON:
                raise RuntimeError(
                    "The selected Arducam config is not monochrome. Use the "
                    "MT9V022_MONO_8b_640x480 configuration."
                )
            print(
                "Opened Arducam USB camera: "
                f"index={camera_index}, {returned_cfg['u32Width']}x"
                f"{returned_cfg['u32Height']}, "
                f"{returned_cfg['u8PixelBits']}-bit"
            )
        except Exception:
            self.sdk.Py_ArduCam_close(self.handle)
            self.handle = None
            raise

    def _apply_configuration(self, config: Any) -> None:
        usb_version = self.camera_cfg["usbType"]
        for index in range(config.configs_length):
            item = config.configs[index]
            item_type = item.type
            usb_selector = (item_type >> 16) & 0xFF
            if usb_selector not in (0, usb_version):
                continue

            base_type = item_type & 0xFFFF
            if base_type == self.config_parser.CONFIG_TYPE_REG:
                ret = self.sdk.Py_ArduCam_writeSensorReg(
                    self.handle, item.params[0], item.params[1]
                )
                if ret not in (None, 0):
                    raise RuntimeError(
                        "MT9V022 register configuration failed: "
                        f"{arducam_error(ret)}"
                    )
            elif base_type == self.config_parser.CONFIG_TYPE_DELAY:
                time.sleep(float(item.params[0]) / 1000.0)
            elif base_type == self.config_parser.CONFIG_TYPE_VRCMD:
                ret = self.sdk.Py_ArduCam_setboardConfig(
                    self.handle,
                    item.params[0],
                    item.params[1],
                    item.params[2],
                    item.params[3],
                    item.params[4 : item.params_length],
                )
                if ret not in (None, 0):
                    raise RuntimeError(
                        "USB Camera Shield configuration failed: "
                        f"{arducam_error(ret)}"
                    )

        ret = self.sdk.Py_ArduCam_registerCtrls(
            self.handle, config.controls, config.controls_length
        )
        if ret not in (None, 0):
            raise RuntimeError(
                f"Arducam control registration failed: {arducam_error(ret)}"
            )

    def start(self) -> None:
        if self.running:
            return
        ret = self.sdk.Py_ArduCam_setMode(
            self.handle, self.sdk.CONTINUOUS_MODE
        )
        if ret not in (None, 0):
            raise RuntimeError(f"Failed to set camera mode: {arducam_error(ret)}")
        ret = self.sdk.Py_ArduCam_beginCaptureImage(self.handle)
        if ret != 0:
            raise RuntimeError(f"Failed to begin capture: {arducam_error(ret)}")

        self.capture_error = None
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="arducam-capture", daemon=True
        )
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        while self.running:
            ret = self.sdk.Py_ArduCam_captureImage(self.handle)
            if ret > 0xFF:
                self.capture_error = RuntimeError(
                    f"Arducam capture failed: {arducam_error(ret)}"
                )
                self.running = False
                with self.signal:
                    self.signal.notify_all()
                return
            if ret > 0:
                with self.signal:
                    self.signal.notify_all()

    def read(self, timeout: float = 1.5) -> np.ndarray:
        """Return the newest available MONO frame and discard older frames."""

        deadline = time.monotonic() + timeout
        while self.sdk.Py_ArduCam_availableImage(self.handle) <= 0:
            if self.capture_error is not None:
                raise self.capture_error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for an Arducam frame. Check USB access, "
                    "the MT9V022 config file, and camera power."
                )
            with self.signal:
                self.signal.wait(remaining)

        latest = None
        queued = max(1, self.sdk.Py_ArduCam_availableImage(self.handle))
        for frame_number in range(queued):
            ret, data, cfg = self.sdk.Py_ArduCam_readImage(self.handle)
            try:
                if ret != 0 or cfg["u32Size"] == 0:
                    raise RuntimeError(
                        f"Failed to read Arducam image: {arducam_error(ret)}"
                    )
                if frame_number == queued - 1:
                    latest = self._mono_to_numpy(data, cfg)
            finally:
                self.sdk.Py_ArduCam_del(self.handle)

        if latest is None:
            raise RuntimeError("Arducam returned no usable image")
        return latest

    def _mono_to_numpy(self, data: Any, cfg: dict) -> np.ndarray:
        width = int(cfg["u32Width"])
        height = int(cfg["u32Height"])
        pixel_count = width * height
        if cfg["u8PixelBytes"] == 1:
            image = np.frombuffer(data, dtype=np.uint8, count=pixel_count)
        elif cfg["u8PixelBytes"] == 2:
            raw = np.frombuffer(data, dtype=np.uint16, count=pixel_count)
            shift = int(cfg["u8PixelBits"]) - 8
            image = (raw >> shift).astype(np.uint8)
        else:
            raise RuntimeError(
                f"Unsupported Arducam pixel width: {cfg['u8PixelBytes']} bytes"
            )
        return image.reshape(height, width).copy()

    def stop(self) -> None:
        if not self.running and self.capture_thread is None:
            return
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        if self.handle is not None:
            self.sdk.Py_ArduCam_endCaptureImage(self.handle)

    def close(self) -> None:
        self.stop()
        if self.handle is not None:
            self.sdk.Py_ArduCam_close(self.handle)
            self.handle = None


def infer_model_hw(input_shape: Sequence[int]) -> Tuple[int, int]:
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Expected a batch-1 4D DPU input, got {shape}")
    if shape[1] == 1:
        return shape[2], shape[3]
    if shape[-1] == 1:
        return shape[1], shape[2]
    raise ValueError(f"Expected a one-channel DPU input, got {shape}")


def prepare_model_input(
    image: np.ndarray, target_height: int, target_width: int, fit: str
) -> np.ndarray:
    import cv2

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected an 8-bit monochrome image, got {image.shape}/{image.dtype}"
        )

    source_height, source_width = image.shape
    if (source_height, source_width) == (target_height, target_width):
        result = image
    elif fit == "strict":
        raise ValueError(
            f"Camera is {source_width}x{source_height}; "
            f"DPU input is {target_width}x{target_height}"
        )
    elif fit == "center-crop":
        if source_height < target_height or source_width < target_width:
            raise ValueError("Camera image is too small for center cropping")
        top = (source_height - target_height) // 2
        left = (source_width - target_width) // 2
        result = image[top : top + target_height, left : left + target_width]
    elif fit == "resize":
        result = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    else:
        raise ValueError(f"Unsupported frame-fit mode: {fit}")
    return np.ascontiguousarray(result)


def load_pipeline() -> SimpleNamespace:
    import main_CNN_Star_Tracker as pipeline

    return SimpleNamespace(
        A_star=pipeline.A_star,
        SVD_method=pipeline.SVD_method,
        draw_centroids=pipeline.draw_centroids,
        geometric_voting=pipeline.geometric_voting,
        run_neural_net=pipeline.run_neural_net,
        visualize_angular_distance=pipeline.visualize_angular_distance,
    )


def load_neural_net(overlay_path: str, xmodel_path: str) -> dict:
    from pynq_dpu import DpuOverlay

    overlay = DpuOverlay(overlay_path)
    overlay.load_model(xmodel_path)
    runner = overlay.runner
    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    return {
        "overlay": overlay,
        "runner": runner,
        "in_shape": input_tensor.dims,
        "out_shape": output_tensor.dims,
        "in_fix": input_tensor.get_attr("fix_point"),
        "out_fix": output_tensor.get_attr("fix_point"),
    }


def process_frame(
    image: np.ndarray,
    args: argparse.Namespace,
    pipeline: SimpleNamespace,
    neural_net: Optional[dict],
    voting: Any,
) -> Tuple[np.ndarray, np.ndarray, list, Tuple[float, float, float], bool]:
    import cv2

    centroids = pipeline.run_neural_net(image.copy(), neural_net)

    voting.star_vectors_ver = []
    phi = theta = psi = 0.0
    solved = False
    if 0 < len(centroids) < 30:
        angular_view, angular_distances, combinations, body_vectors = (
            pipeline.visualize_angular_distance(centroids, image, None)
        )
        angular_view = angular_view.astype(np.uint8)
        voting.centroid_result = centroids
        voting.star_identification(angular_distances, combinations)
        attitude_matrix, solved = pipeline.SVD_method(
            body_vectors, voting.star_vectors_ver
        )
        if solved:
            attitude_matrix = pipeline.A_star @ attitude_matrix
            phi = math.degrees(
                math.atan2(attitude_matrix[0, 1], attitude_matrix[0, 0])
            )
            theta = math.degrees(
                math.atan2(
                    -attitude_matrix[0, 2],
                    math.sqrt(max(0.0, 1.0 - attitude_matrix[0, 2] ** 2)),
                )
            )
            psi = math.degrees(
                math.atan2(attitude_matrix[1, 2], attitude_matrix[2, 2])
            )
    else:
        angular_view = np.zeros((*image.shape, 3), dtype=np.uint8)

    display_image = cv2.cvtColor(
        np.clip(image.astype(np.float32) * args.display_gain, 0, 255).astype(
            np.uint8
        ),
        cv2.COLOR_GRAY2BGR,
    )
    pipeline.draw_centroids(
        centroids,
        display_image,
        1.0,
        args.centroid_radius,
        voting.star_vectors_ver,
        voting.star_catalog,
    )
    return display_image, angular_view, centroids, (phi, theta, psi), solved


def save_results(
    output_dir: str,
    mode: str,
    attitude: list,
    identification_rate: list,
    animation: list,
    save_video: bool,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"usb_camera_{mode}_{timestamp}"
    np.save(directory / f"attitude_{stem}.npy", np.asarray(attitude))
    np.save(directory / f"id_rate_{stem}.npy", np.asarray(identification_rate))
    if save_video:
        np.save(directory / f"animation_{stem}.npy", np.asarray(animation))


def main_camera(args: argparse.Namespace) -> None:
    import cv2

    pipeline = load_pipeline()
    print(f"Loading DPU overlay: {args.overlay}")
    print(f"Loading XModel: {args.xmodel}")
    neural_net = load_neural_net(args.overlay, args.xmodel)
    model_height, model_width = infer_model_hw(neural_net["in_shape"])
    print(f"DPU input: {model_width}x{model_height} MONO8")

    print(f"Opening Arducam USB camera with config: {args.camera_config}")

    camera = ArducamUsbCamera(args.camera_config, args.camera_index)
    voting = pipeline.geometric_voting()
    attitude = []
    identification_rate = []
    animation = []
    frame_index = -1
    start_time = time.perf_counter()

    try:
        camera.start()
        while args.frames == 0 or frame_index + 1 < args.frames:
            frame_index += 1
            captured = camera.read(args.capture_timeout)
            if frame_index == 0:
                print(
                    "Captured first Arducam USB frame: "
                    f"{captured.shape[1]}x{captured.shape[0]} "
                    f"{captured.dtype}; sending frames to the DPU"
                )
            image = prepare_model_input(
                captured, model_height, model_width, args.frame_fit
            )
            display_image, angular_view, centroids, euler, solved = process_frame(
                image, args, pipeline, neural_net, voting
            )

            identified = (
                np.asarray(voting.star_vectors_ver)[:, 4]
                if len(voting.star_vectors_ver) > 0
                else []
            )
            identification_rate.append([len(centroids), len(identified)])
            if solved:
                attitude.append([frame_index, *euler])
            if args.save_video:
                animation.append(display_image)

            elapsed = time.perf_counter() - start_time
            average_fps = (frame_index + 1) / elapsed if elapsed else 0.0
            print(
                "frame={}, centroids={}, ids={}, "
                "att=[{:.4f},{:.4f},{:.4f}], avg_fps={:.2f}".format(
                    frame_index,
                    len(centroids),
                    len(identified),
                    *euler,
                    average_fps,
                )
            )

            if args.display:
                cv2.imshow("Star Detection and Centroiding", display_image)
                cv2.imshow("Angular Distance", angular_view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving partial results")
    finally:
        camera.close()
        if args.display:
            cv2.destroyAllWindows()
        save_results(
            args.output_dir,
            args.mode,
            attitude,
            identification_rate,
            animation,
            args.save_video,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MT9V022 + Arducam USB Shield + DPU star tracker"
    )
    parser.add_argument(
        "--camera-config",
        required=True,
        help="Arducam MT9V022 MONO8 .cfg file",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-timeout", type=float, default=1.5)
    parser.add_argument(
        "--mode",
        choices=("NN",),
        default="NN",
        help="DPU neural-network inference (the only supported camera mode)",
    )
    parser.add_argument("--overlay", default="dpu.bit")
    parser.add_argument(
        "--xmodel", default="DPU_models/CNN_Star_Tracker_Model_V6.xmodel"
    )
    parser.add_argument(
        "--frame-fit",
        choices=("strict", "center-crop", "resize"),
        default="strict",
    )
    parser.add_argument("--model-width", type=int, default=640)
    parser.add_argument("--model-height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=0, help="0: run continuously")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-gain", type=float, default=3.0)
    parser.add_argument("--centroid-radius", type=int, default=15)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default="saved_results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 0:
        raise SystemExit("--frames must be non-negative")
    if args.capture_timeout <= 0:
        raise SystemExit("--capture-timeout must be positive")
    main_camera(args)


if __name__ == "__main__":
    main()
