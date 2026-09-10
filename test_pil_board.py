"""PC tests of the copied board adapter. No physical DPU or USB is exercised."""
import copy
import json
import socket
import struct
import threading
import unittest
import zlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import main_CNN_Star_Tracker_PIL as board

PROFILE = dict(width=12, height=8, focal_px=20., cx=6., cy=4.,
               calibration_id="test", distortion="none", pixel_convention="simulator_native",
               centroid_offset_px=[0., 0.])
POINTS = np.array([[2., 2.], [8., 3.], [5., 6.]])


class Matcher:
    def star_identification(self, angles, pairs):
        # Independent inertial catalog for an identity simulator camera pose.
        self.star_vectors_ver = []
        for i, (u, v) in enumerate(POINTS):
            direction = np.array([u-6., v-4., 20.])
            direction /= np.linalg.norm(direction)
            self.star_vectors_ver.append([*direction, i, i, 3, 100+i])


class AdapterTests(unittest.TestCase):
    def tracker(self, detector=lambda image: POINTS):
        return board.Tracker(copy.deepcopy(PROFILE), detector, Matcher())

    def test_known_pose_and_clear_stale_result(self):
        tracker = self.tracker()
        image = np.zeros((8, 12), dtype=np.uint8)
        result = tracker.process_one_frame(image)
        self.assertEqual(result["status"], "ok")
        np.testing.assert_allclose(np.reshape(result["rotation_row_major"], (3, 3)),
                                   [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
        self.assertEqual(len(result["identified"]), 3)
        tracker.detector = lambda image: []
        result = tracker.process_one_frame(image)
        self.assertEqual(result["status"], "no_solution")
        self.assertNotIn("rotation_row_major", result)
        self.assertEqual(tracker.matcher.star_vectors_ver, [])

    def test_tensor_layout_and_preprocessing(self):
        class Runner:
            calls = 0
            def execute_async(self, inputs, outputs):
                self.calls += 1
                self.input = inputs[0].copy()
                outputs[0].fill(0)
                return 123
            def wait(self, jid):
                assert jid == 123
        for layout, ishape, oshape in [("NHWC", (1, 8, 12, 1), (1, 8, 12, 2)),
                                       ("NCHW", (1, 1, 8, 12), (1, 2, 8, 12))]:
            self.assertEqual(board.tensor_layout(ishape, 1), (layout, 8, 12))
            runner = Runner()
            ctx = dict(runner=runner, height=8, width=12, in_shape=ishape,
                       out_shape=oshape, in_fix=5, out_fix=3, out_layout=layout)
            image = np.arange(96, dtype=np.uint8).reshape(8, 12)
            self.assertEqual(board.run_neural_net(image, ctx), [])
            expected = np.clip(np.round((image.astype(np.float64)-25.36114133)/44.31162568*32), -128, 127).astype(np.int8)
            np.testing.assert_array_equal(runner.input.ravel(), expected.ravel())
            with self.assertRaises(ValueError):
                board.run_neural_net(image.T, ctx)
        with self.assertRaises(ValueError):
            board.tensor_layout((1, 2, 12, 2), 2)

    def exchange(self, change=None, payload=None, detector=None, hello_change=None, truncated=False):
        client, server = socket.socketpair()
        tracker = self.tracker() if detector is None else self.tracker(detector)
        record = {}
        def worker():
            with server:
                try:
                    record["result"] = board.handle_connection(server, tracker, {"device": "test"}, .25)
                except Exception as exc:
                    record["error"] = exc
        thread = threading.Thread(target=worker)
        thread.start()
        hello = dict(type="HELLO", version=1, run_id="run", pixel_format="GRAY8",
                     **{k: v for k, v in PROFILE.items() if k != "centroid_offset_px"})
        hello.update(hello_change or {})
        raw = bytes(range(96))
        header = dict(type="IMAGE", version=1, run_id="run", frame_id=7, width=12, height=8,
                      pixel_format="GRAY8", payload_bytes=96,
                      image_crc32="{:08x}".format(zlib.crc32(raw) & 0xffffffff))
        header.update(change or {})
        try:
            with client:
                board.send_json(client, hello, 1.)
                if not hello_change:
                    ready = board.recv_json(client, board.time.monotonic()+1)
                    self.assertEqual(ready["type"], "READY")
                    data = json.dumps(header).encode()
                    # Fragment the header prefix and concatenate JSON with image data.
                    prefix = struct.pack("!I", len(data))
                    client.sendall(prefix[:1])
                    client.sendall(prefix[1:]+data+(raw if payload is None else payload))
                    if not truncated:
                        try:
                            record["reply"] = board.recv_json(client, board.time.monotonic()+1)
                        except (EOFError, ConnectionResetError):
                            pass
        finally:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        return record

    def test_fragmented_roundtrip(self):
        record = self.exchange()
        self.assertNotIn("error", record)
        self.assertEqual(record["reply"]["status"], "ok")
        self.assertEqual(record["reply"]["frame_id"], 7)

    def test_bad_crc_never_runs_detector(self):
        def detector(image):
            self.fail("CRC failure must not reach inference")
        record = self.exchange(payload=bytes(96), detector=detector)
        self.assertEqual(record["reply"]["status"], "input_error")

    def test_size_id_and_calibration_rejection(self):
        for change in [{"payload_bytes": 95}, {"frame_id": True}, {"run_id": "wrong"}, {"width": 8}]:
            self.assertIsInstance(self.exchange(change=change)["error"], ValueError)
        self.assertIsInstance(self.exchange(hello_change={"focal_px": 21.})["error"], ValueError)

    def test_truncated_and_processing_failure(self):
        self.assertIsInstance(self.exchange(payload=b"", truncated=True)["error"], EOFError)
        def broken(image):
            raise RuntimeError("fake inference failure")
        result = self.exchange(detector=broken)["reply"]
        self.assertEqual(result["status"], "input_error")
        self.assertNotIn("rotation_row_major", result)

    def test_border_centroid_does_not_wrap(self):
        seg = np.zeros((8, 12))
        seg[0, 0] = seg[-1, -1] = 1
        self.assertEqual(board.trilateration_centroid_vectorization(np.zeros_like(seg), seg, 1, 1), [])

    @unittest.skipUnless(Path('/home/eslab/miniconda3/Simplified-Verification-Platform-for-Star-Trackers/bin/verification_platform').exists(), "WSL C++ host not available")
    def test_existing_cpp_host(self):
        root = Path('/home/eslab/miniconda3/Simplified-Verification-Platform-for-Star-Trackers')
        sequence = iter([POINTS, []])
        tracker = self.tracker(lambda image: next(sequence))
        failures = []
        with socket.socket() as listener, tempfile.TemporaryDirectory(prefix="board_pil_") as temp:
            listener.bind(('127.0.0.1', 0))
            listener.listen(2)
            listener.settimeout(10)
            def worker():
                try:
                    for _ in range(2):
                        conn, _ = listener.accept()
                        with conn:
                            board.handle_connection(conn, tracker, {"device": "PC test fixture, no DPU"}, 3)
                except Exception as exc:
                    failures.append(exc)
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            config = dict(mode="tcp", host="127.0.0.1", port=listener.getsockname()[1],
                          output=str(Path(temp)/"run"), simulator_config="config/common.yml",
                          count=2, timeout_ms=3000, threshold_arcsec=15,
                          attitudes_wxyz=[[1, 0, 0, 0], [1, 0, 0, 0]],
                          **{k: PROFILE[k] for k in ("width", "height", "focal_px", "cx", "cy", "calibration_id")})
            config_path = Path(temp)/"config.json"
            config_path.write_text(json.dumps(config))
            process = subprocess.run([str(root/'bin/verification_platform'), '--pil', str(config_path)],
                                     cwd=root, capture_output=True, text=True, timeout=15)
            thread.join(12)
            self.assertFalse(thread.is_alive())
            self.assertFalse(failures, failures)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            summary = json.loads((Path(temp)/"run/summary.json").read_text())
            self.assertEqual(summary['counts'], {'correct': 1, 'no_solution': 1})


if __name__ == "__main__":
    unittest.main()
