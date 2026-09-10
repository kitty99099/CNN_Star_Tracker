"""Ultra96 PIL variant copied from main_CNN_Star_Tracker.py.

Original video helpers are retained; the entry point serves PIL TCP protocol v1.
"""
import numpy as np
#from camera_software import camera_software
import cv2
from itertools import combinations

from scipy.linalg import lstsq

import time 
import sys

import argparse

import math

import os 
import json
import socket
import struct
import zlib
import hashlib
from pathlib import Path

MAX_JSON = 1024 * 1024
MAX_IMAGE = 16 * 1024 * 1024
SCRIPT_PATH = Path(__file__).resolve()

# rotate star tracker body frame 
from math import sin, cos, radians
phi = radians(0) # Z
theta = radians(-90) # Y
psi = radians(-90) # X
A_star = np.array([  [ cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta) ],
                    [ -cos(psi)*sin(phi) + sin(psi)*sin(theta)*cos(phi), cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi), sin(psi)*cos(theta) ],
                    [ sin(psi)*sin(phi) + cos(psi)*sin(theta)*cos(phi), -sin(psi)*cos(phi) + cos(psi)*sin(theta)*sin(phi), cos(psi)*cos(theta) ]  ])

def SVD_method(star_vectors, star_vectors_ver):
    """
        "Markley, Landis. (1987). Attitude Determination Using Vector Observations and the Singular Value Decomposition. J. Astronaut. Sci.. 38."  
        @param star_vectors: a list saving computed body frame star vectors from centroid points. row element [ s_x, s_y, s_z ]
        @param star_vectors_ver:  a list matched J2000 ECI frame catalog star vectors from centroid points. verified. row element [ x, y, z, catalog_id, centroid_id, votes, hip id, magnitude ].  
        @return A:  rotation from ICRF to body frame. is an empty list if not solution is availale
        @return solution: True is solution exist
    """

    # number of identified stars
    number = len(star_vectors_ver)

    if number >= 3 :

        shape = (3, 3)
        B = np.zeros(shape)
        mag_total = 0
        for row in range(number):
            id = int(star_vectors_ver[row][4]) #centroid id
            # body frame
            x_b = star_vectors[id,0]
            y_b = star_vectors[id,1]
            z_b = star_vectors[id,2]
            # inertial frame
            x_i = star_vectors_ver[row][0]
            y_i = star_vectors_ver[row][1] 
            z_i = star_vectors_ver[row][2]
            # # magnitude
            # mag = star_vectors_ver[row][7]
            # mag_total = mag_total + mag
            
            bi = np.array( [ x_b, y_b, z_b ] )
            ri = np.array( [ x_i, y_i, z_i ] )
        
            #Bi = mag * np.outer(bi, ri) # 3x1 @ 1x3 = 3x3, weighted by magnitude
            Bi = np.outer(bi, ri) # 3x1 @ 1x3 = 3x3
            B = B + Bi
        # normalize weight
        #B = B * (1/mag_total)

        # SVD
        u, s, vT = np.linalg.svd( B )
        v = np.transpose(vT)
        detU = np.linalg.det(u)
        detV = np.linalg.det(v)
        d = detU*detV
        diag = np.eye(3)
        diag[2,2] = d
        A = u@diag@vT
        solution = True

    else:
        A = []
        solution = False

    return A, solution

def distortion_correction(centroids_distorted, p, u_0, v_0):
    """
        @param centroids_distorted: list of distorted centroids
        @param p: distortion coefficients
        @param u_0:
        @param v_0:
        Ref:
        Heikkila, J., and Silven, O., "A Four-Step Camera Calibration Procedure with Implicit
        Image Correction," Proceedings of IEEE Computer Society Conference on Computer Vision
        and Pattern Recognition, San Juan, Puerto Rico, USA, 1997
    """
    a_1 = p[0]
    a_2 = p[1]
    a_3 = p[2]
    a_4 = p[3]
    a_5 = p[4]
    a_6 = p[5]
    a_7 = p[6]
    a_8 = p[7]

    correctd_centroid_result = []

    for item in centroids_distorted:
        u = item[0]
        v = item[1]

        u_d = u - u_0
        v_d = v - v_0

        r = math.sqrt(u_d**2 + v_d**2)
        G = (a_5*r**2 + a_6*u_d + a_7*v_d + a_8)*r**2 + 1
        u_u = (1/G) * (u_d + u_d*(a_1*r**2 + a_2*r**4) + 2*a_3*u_d*v_d + a_4*(r**2+2*u_d**2))
        v_u = (1/G) * (v_d + v_d*(a_1*r**2 + a_2*r**4) + a_3*(r**2+2*v_d**2) + 2*a_4*u_d*v_d)

        correctd_centroid_result.append([u_u + u_0, v_u + v_0])

    return correctd_centroid_result

def visualize_angular_distance(centroid, img, selected_stars=None):

    ### calibration parameters ###
    # # test system 
    # focal_length = 2.68674091e+03 
    # u0 = 3.43471001e+02 
    # v0 = 2.25697237e+02
    
    # night sky test March 11
    # focal_length = 2.67643071e+03
    # u0 = 3.53793375e+02
    # v0 = 1.99364958e+02

    # night sky test March 11 Matlab Cal
    focal_length = 2680.049125
    u0 = 352.84399
    v0 = 199.46077
    p = [ 4.72251629e-06, -1.47548083e-11,  5.40718521e-10,  3.15303463e-10, -1.47546930e-11,  1.26044011e-15, -2.20571693e-14, 4.72252718e-06]
    ######


    background = np.zeros(( img.shape[0], img.shape[1], 3))
    radius = 10 # star circle radius
    red_bgr = (0, 0, 255)
    green_bgr = (0, 255, 0)
    yellow_bgr = (0, 255, 255)
    white_bgr = (255, 255, 255)
    line_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    font_thickness = 1

    centroid = distortion_correction(centroid, p, u0, v0)
    centroid = np.asarray(centroid)
    #distortion correction
    #k1, k2, k3, p1, p2 = 2.05948283e-01, 1.00197616e+01, 6.27783624e-01, 2.19815443e-04, -1.81622608e-03 # test system
    #k1, k2, p1, p2, k3 = 2.02304236e-01,  1.19154080e+01, -7.08025042e-03, -1.24185114e-03, -2.40661160e+02 # night sky test March 11
    # k1, k2, p1, p2, k3 = 0.23801, 6.40623, -0.00699, -0.00156, 0.00000
    # for i in centroid:
    #     i[0] = (i[0]-u0)/focal_length
    #     i[1] = (i[1]-v0)/focal_length
    #     r = np.sqrt(i[0]**2+i[1]**2)
    #     k_inv = 1/(1+k1*r**2+k2*r**4+k3*r**6)
    #     delta_x = 2*p1*i[0]*i[1]+p2*(r**2+2*i[0]**2)
    #     delta_y = p1*(r**2+2*i[1]**2)+2*p2*i[0]*i[1]
    #     i[0] = ((i[0]-delta_x)*k_inv)*focal_length+u0
    #     i[1] = ((i[1]-delta_y)*k_inv)*focal_length+v0

    if selected_stars is not None:
        centroid = centroid[selected_stars]
    centroid_uv_int = centroid[:,0:2].astype('int')

    # get body frame star vectors
    star_vector_array = np.column_stack((-centroid[:,0:2], np.zeros(centroid.shape[0])))
    star_vector_array = star_vector_array + np.array([u0, v0, focal_length])

    # normalize star vectors 
    row_sum = np.sum(star_vector_array*star_vector_array, axis=1, keepdims=True)
    row_sum = np.sqrt(row_sum)
    star_vector_array = star_vector_array / row_sum

    # draw lines and angular distance
    dot_product_array = star_vector_array @ np.transpose(star_vector_array)
    dot_product_array = np.clip(dot_product_array, -1.0, 1.0)
    angular_distance_array = np.rad2deg(np.arccos(dot_product_array))

    indices = np.arange(angular_distance_array.shape[0])
    comb = np.asarray(list(combinations(indices, 2)))
    for i in range(comb.shape[0]):
        id1 = comb[i,0]
        id2 = comb[i,1]
        angular_distance = angular_distance_array[id1, id2]
        text = '{:.4f} deg'.format(angular_distance)
        centroid_1 = centroid_uv_int[id1]
        centroid_2 = centroid_uv_int[id2]
        midpoint = np.abs((centroid_2 + centroid_1)/2).astype('int')
        cv2.line(background, tuple(centroid_1), tuple(centroid_2), green_bgr, line_thickness)
        cv2.putText(background, text, tuple(midpoint), font, fontscale, white_bgr, font_thickness, cv2.LINE_AA)
        
    # draw stars 
    for i in range(centroid.shape[0]):
        cv2.circle(background, tuple(centroid_uv_int[i]), radius, red_bgr, -1 )
        text = f'{i}'
        cv2.putText(background, text, tuple(centroid_uv_int[i]), font, fontscale*2, yellow_bgr, font_thickness*2, cv2.LINE_AA)


    #cv2.imwrite('angular_distance_vis.png', background)

    return background, angular_distance_array, comb, star_vector_array

def trilateration_centroid_vectorization(dist_map, seg_map, radius, pixel_size=6/1000.0):
    """
        use points within a 5x5 (at least 5x5) window to determine centroid  
        https://www3.nd.edu/~cpoellab/teaching/cse40815/Chapter10.pdf
        @param dist_map: distance map.
        @param seg_map: segmentation map.   
        @param pixel_size: in mm
        @param radius: radius of the window size, in pixels
        @return centroid_est: estimated centroids. a list of [u,v]. in mm
    """
    dist_map = dist_map.astype('float64')
    seg_map = seg_map.astype('float64')
    image_height = dist_map.shape[0]
    image_width = dist_map.shape[1] 
    centroid_est = []
    threshold = 2 #0.5*sqrt(2) 
    
    coarse_centroid = np.multiply((dist_map <= threshold),  (seg_map > 0))
    row_list, col_list = np.nonzero(coarse_centroid)

    for current_row, current_col in zip(row_list, col_list):

        if seg_map[current_row, current_col] > 0  :

            x_i = []
            y_i = []
            r_i = []
            for row_shift in range (-radius, radius+1, 1): 
                for col_shift in range(-radius, radius+1, 1):
                    row = current_row + row_shift
                    col = current_col + col_shift 
                    try:
                        if 0 <= row < image_height and 0 <= col < image_width and seg_map[row, col] > 0:
                            u = (col + 0.5) 
                            v = (row + 0.5) 
                            r_i.append(dist_map[row, col])
                            x_i.append(u)
                            y_i.append(v)
                            seg_map[row, col] = 0 # avoid getting reused
                    except IndexError:
                        pass
            n = len(x_i)
            if n < 3:
                continue
            x_n = x_i[-1]
            y_n = y_i[-1]
            r_n = r_i[-1]

            # create matrix A
            A = np.zeros((n-1, 2))
            for i in range(n-1):
                A[i, 0] = 2 * (x_n - x_i[i])
                A[i, 1] = 2 * (y_n - y_i[i])

            # create matrix B
            B = np.zeros(n-1)
            for i in range(n-1):
                B[i] = r_i[i]**2 - r_n**2 - x_i[i]**2 - y_i[i]**2 + x_n**2 + y_n**2

            # # caluclate centroid
            # A_T = np.transpose(A)
            # A_TA = A_T@A
            # if np.linalg.cond(A_TA) < 1/sys.float_info.epsilon: # to abandon singular matrix
            #     x = np.linalg.inv(A_TA) @ A_T @ B
            #     u = x[0] * pixel_size
            #     v = x[1] * pixel_size
            #     centroid_est.append([u, v])

            # faster centroid calculation
            # https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
            x = lstsq(A, B, lapack_driver='gelsy') # sloving Ax=b
            if x[2] < 2 or not np.isfinite(x[0]).all():
                continue
            u = x[0][0] * pixel_size
            v = x[0][1] * pixel_size
            centroid_est.append([u, v])

            if len(centroid_est) > 50: # too many detection
                return centroid_est  

    return centroid_est

def init_neural_net(xmodel_path, bitfile="dpu.bit"):
    from pynq_dpu import DpuOverlay
    overlay = DpuOverlay(bitfile)
    overlay.load_model(xmodel_path)      # 你編好的 xmodel
    runner = overlay.runner

    #取得 tensor 屬性
    inputs, outputs = runner.get_input_tensors(), runner.get_output_tensors()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("This copied inference path requires one input and one output tensor")
    inp_t = inputs[0]
    out_t = outputs[0]
    in_shape = tuple(int(x) for x in inp_t.dims)
    out_shape = tuple(int(x) for x in out_t.dims)
    in_layout, height, width = tensor_layout(in_shape, 1)
    out_layout, out_h, out_w = tensor_layout(out_shape, 2)
    if (out_h, out_w) != (height, width):
        raise ValueError("Output maps must have the same H/W as the image")
    for tensor in (inp_t, out_t):
        dtype = str(tensor.dtype).lower()
        if dtype not in ("int8", "xint8"):
            raise ValueError("This copied inference path requires INT8 tensors, got " + dtype)
    return {
        "overlay": overlay,
        "runner": runner,
        "in_shape": in_shape,
        "out_shape": out_shape,
        "in_layout": in_layout,
        "out_layout": out_layout,
        "height": height,
        "width": width,
        "in_fix": inp_t.get_attr("fix_point"),
        "out_fix": out_t.get_attr("fix_point"),
    }


def tensor_layout(shape, channels):
    if len(shape) != 4 or shape[0] != 1 or any(x <= 0 for x in shape):
        raise ValueError("Expected a positive batch-one 4D tensor: " + str(shape))
    layouts = []
    if shape[-1] == channels:
        layouts.append(("NHWC", shape[1], shape[2]))
    if shape[1] == channels:
        layouts.append(("NCHW", shape[2], shape[3]))
    if len(layouts) != 1:
        raise ValueError("Unsupported or ambiguous tensor layout: " + str(shape))
    return layouts[0]

def run_neural_net(img, nn_ctx):

    pixel_size = 1 # get centroid in pixel instead of mm
    radius = 7

    # for data_Oct19
    mean = [25.36114133]
    std =  [44.31162568]

    runner = nn_ctx["runner"]
    if img.ndim != 2 or img.shape != (nn_ctx["height"], nn_ctx["width"]):
        raise ValueError("Image size does not match the loaded model; no implicit resize")
    in_shape = nn_ctx["in_shape"]
    out_shape = nn_ctx["out_shape"]
    in_fix = nn_ctx["in_fix"]
    out_fix = nn_ctx["out_fix"]

    # img_np: (H,W) float32/float64
    # 前處理：normalize + 量化成 int8
    img_f = img.astype(np.float32)
    img_f = (img_f - mean) / std
    scale = 2 ** in_fix
    inp_int8 = np.clip(np.round(img_f * scale), -128, 127).astype(np.int8).reshape(in_shape)

    # 配置 buffer
    input_buf = [np.empty_like(inp_int8)]
    output_buf = [np.empty(out_shape, dtype=np.int8)]
    input_buf[0][...] = inp_int8

    # 執行
    jid = runner.execute_async(input_buf, output_buf)
    status = runner.wait(jid)
    if status not in (None, 0):
        raise RuntimeError("DPU execution failed: " + str(status))

    # 反量化輸出
    out_scale = 2 ** out_fix
    out_f = output_buf[0].astype(np.float32) / out_scale
    if nn_ctx["out_layout"] == "NCHW":
        seg_prediction = (1.0 / (1.0 + np.exp(-out_f[0, 0]))) > 0.5
        dist_prediction = out_f[0, 1]
    elif nn_ctx["out_layout"] == "NHWC":
        seg_prediction = (1.0 / (1.0 + np.exp(-out_f[0, :, :, 0]))) > 0.5
        dist_prediction = out_f[0, :, :, 1]
    else:
        raise ValueError(f"Unsupported DPU output shape: {out_f.shape}")

    centroid_est = trilateration_centroid_vectorization(dist_prediction.copy(), seg_prediction.copy(), radius, pixel_size)

    return centroid_est

def draw_centroids(centroid, draw_background, pixel_size, radius, star_vectors_ver, star_catalog):
    """
        @param centroid: real or estimated centroids.  
        @param draw_background
        @param pixel_size: in mm.  
        @param radius: radius of the drawn circle in pixels, should be the same as the window size of the centroid method
        @param star_vectors_ver: row element [ x, y, z, catalog_id, centroid_id, votes ]
    """
    Green = (0,255,0) # RGB
    white_bgr = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.25
    font_thickness = 1
    
    if len(star_vectors_ver) > 0:
        catalog_id = np.asarray(star_vectors_ver)[:,3]            
        centroid_id = np.asarray(star_vectors_ver)[:,4]            
    else:
        catalog_id = []
        centroid_id = []

    for i in range(len(centroid)):

        u = centroid[i][0]
        v = centroid[i][1]
        
        # which pixel the star centroid is located
        u_p = int(u // pixel_size)
        v_p = int(v // pixel_size)

        center = (u_p, v_p)
        thickness = 1 # Using thickness of -1 px to fill the circle
        if i in centroid_id:
            j = list(centroid_id).index(i)
            id = int(catalog_id[j])
            cv2.putText(draw_background, f'HIP={int(star_catalog[id][4])},M={star_catalog[id][3]}', (u_p, v_p-20), font, fontscale, white_bgr, font_thickness, cv2.LINE_AA)
        cv2.circle(draw_background, center, radius, Green, thickness)

def main_video(args):
    from geometric_voting import geometric_voting
    from tqdm import trange
    detection_vis_radius = 15

    geometric_voting_obj = geometric_voting()
    phi = 0 # initialize attitude determination
    theta = 0
    psi = 0
    attitude = []
    id_rate = []

    # load video 
    video_path = os.path.join('./saved_results', args.video_file)
    video = np.load(video_path)

    # load neural net
    nn_ctx = init_neural_net(args.xmodel) if args.mode == 'NN' else None

    print(f'running {args.mode}')

    animation = []

    i = -1
    try:
        for i in trange(video.shape[0], smoothing=0):

            img = video[i].astype('float32')

            if args.mode == 'NN':
                centroid_est = run_neural_net(img.copy(), nn_ctx)
            elif args.mode == 'baseline':
                centroid_est = run_baseline(img)
            else:
                print("WRONG MODE")
                break

            if len(centroid_est) > 0 and len(centroid_est) < 30:
                vis, angular_distance_array, comb, star_vector_array = visualize_angular_distance(centroid_est, img, None)
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                geometric_voting_obj.centroid_result = centroid_est
                geometric_voting_obj.star_identification(angular_distance_array, comb)
                A, solution = SVD_method(star_vector_array, geometric_voting_obj.star_vectors_ver)
                if solution: # 3-2-1 euler angles
                    A = A_star @ A  # from ICRF, to Z axis boresight body frame, to X axis boresight body frame
                    phi = math.degrees(math.atan2(A[0,1], A[0,0]))
                    theta = math.degrees(math.atan2( -A[0,2], math.sqrt(1-A[0,2]**2) ))
                    psi = math.degrees(math.atan2(A[1,2], A[2,2]))
                    attitude.append([i, phi, theta, psi])
            else:
                vis = np.zeros(( img.shape[0], img.shape[1], 3))

            img = cv2.cvtColor(np.clip(img*3, a_min=0, a_max=255).astype('uint8'), cv2.COLOR_GRAY2RGB)

            draw_centroids(centroid_est, img, 1.0, detection_vis_radius, geometric_voting_obj.star_vectors_ver, geometric_voting_obj.star_catalog)

            identified_centroids = []
            if len(geometric_voting_obj.star_vectors_ver) > 0:
                identified_centroids = np.asarray(geometric_voting_obj.star_vectors_ver)[:,4]
            id_rate.append([len(centroid_est), len(identified_centroids)])

            if args.display:
                cv2.imshow('Star Detection and Centroiding', img)
                cv2.imshow('Angular Distacne', vis)
            animation.append(img)

            print('Num of Centroids = {}, num of ids = {}, att=[{:.4f},{:.4f},{:.4f}]'.format(len(centroid_est), len(geometric_voting_obj.star_vectors_ver), phi, theta, psi))
            # two lines below is to erase the previous print message
            sys.stdout.write("\033[F") # Cursor up one line
            sys.stdout.write("\033[K") # Clear to the end of line ( if you print something shorter than before)

            if args.display:
                k = cv2.waitKey(1)
                if k == ord('q'):
                    print("Close Program")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
    finally:
        if args.display:
            cv2.destroyAllWindows()

    print("save star detection and centroiding results")
    np.save(f'./saved_results/attitude_{args.mode}_{i}.npy', np.asarray(attitude))
    np.save(f'./saved_results/id_rate_{args.mode}_{i}.npy', np.asarray(id_rate))
    np.save(f'./saved_results/animation_{args.mode}_{i}.npy', np.asarray(animation))

def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid " + name)
    return value


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid " + name)
    return float(value)


def load_profile(path):
    with open(path, encoding="utf-8") as stream:
        profile = json.load(stream)
    integer(profile.get("width"), "width", 1, MAX_IMAGE)
    integer(profile.get("height"), "height", 1, MAX_IMAGE)
    if profile["width"] * profile["height"] > MAX_IMAGE:
        raise ValueError("Image exceeds protocol limit")
    for key in ("focal_px", "cx", "cy"):
        finite(profile.get(key), key)
    if profile["focal_px"] <= 0:
        raise ValueError("Focal length must be positive")
    if not isinstance(profile.get("calibration_id"), str) or not profile["calibration_id"]:
        raise ValueError("calibration_id required")
    if profile.get("distortion") != "none" or profile.get("pixel_convention") != "simulator_native":
        raise ValueError("PIL v1 supports ideal simulator-native coordinates only")
    offsets = profile.get("centroid_offset_px")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("Explicit centroid_offset_px [du,dv] required")
    for offset in offsets:
        finite(offset, "centroid offset")
    return profile


def reset_matcher(matcher):
    for key in ("centroid_result", "image_pair", "star_vectors",
                "star_vectors_unver", "star_vectors_ver"):
        setattr(matcher, key, [])


class Tracker:
    """One initialized DPU and catalog, reused for all connections."""

    def __init__(self, profile, detector, matcher):
        self.profile = profile
        self.detector = detector
        self.matcher = matcher

    def process_one_frame(self, image):
        reset_matcher(self.matcher)
        result = {"status": "no_solution", "centroids": [], "identified": []}
        if image.dtype != np.uint8 or image.shape != (self.profile["height"], self.profile["width"]):
            raise ValueError("Expected profile-sized uint8 grayscale image")
        raw = np.asarray(self.detector(image), dtype=np.float64)
        if raw.size == 0:
            result["reason"] = "no_centroids"
            return result
        if raw.ndim != 2 or raw.shape[1] < 2 or not np.isfinite(raw).all():
            raise ValueError("Invalid detector centroids")
        # This explicit profile conversion is applied once to geometry and diagnostics.
        centroids = raw[:, :2] + np.asarray(self.profile["centroid_offset_px"])
        if len(centroids) > 4096:
            raise ValueError("Too many centroid diagnostics")
        result["centroids"] = centroids.tolist()
        # Preserve the original tracker gate (<30 detections); require >=3 for SVD.
        if not 3 <= len(centroids) < 30:
            result["reason"] = "centroid_count_outside_3_to_29"
            return result
        p = self.profile
        vectors = np.column_stack((p["cx"] - centroids[:, 0],
                                   p["cy"] - centroids[:, 1],
                                   np.full(len(centroids), p["focal_px"])))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        angles = np.degrees(np.arccos(np.clip(vectors @ vectors.T, -1., 1.)))
        pairs = np.asarray(list(combinations(range(len(centroids)), 2)))
        self.matcher.centroid_result = centroids.tolist()
        self.matcher.star_identification(angles, pairs)
        verified = self.matcher.star_vectors_ver
        seen_hip, seen_centroid = set(), set()
        for row in verified:
            if not np.isfinite(row[:7]).all() or row[4] != int(row[4]) or row[6] != int(row[6]):
                raise ValueError("Invalid identified star")
            cid = integer(int(row[4]), "centroid ID", 0, len(centroids) - 1)
            hip = integer(int(row[6]), "HIP", 1, 2147483647)
            if hip in seen_hip or cid in seen_centroid:
                result["reason"] = "ambiguous_identification"
                result["identified"] = []
                return result
            seen_hip.add(hip)
            seen_centroid.add(cid)
            result["identified"].append([hip, *centroids[cid].tolist()])
        if len(verified) < 3:
            result["reason"] = "insufficient_identified_stars"
            return result
        if np.linalg.matrix_rank(np.asarray(verified)[:, :3]) < 2:
            result["reason"] = "degenerate_star_vectors"
            return result
        A, solution = SVD_method(vectors, verified)
        if not solution:
            return result
        A = A_star @ A
        if not np.isfinite(A).all() or not np.allclose(A @ A.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(A)-1) > 1e-6:
            raise ValueError("Invalid SVD rotation")
        result.update(status="ok", frame="cnn_inertial_to_body",
                      rotation_row_major=A.ravel().tolist(),
                      euler_zyx_deg=[math.degrees(math.atan2(A[0, 1], A[0, 0])),
                                     math.degrees(math.atan2(-A[0, 2], math.hypot(A[0, 0], A[0, 1]))),
                                     math.degrees(math.atan2(A[1, 2], A[2, 2]))])
        return result


def recv_exact(conn, size, deadline):
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Receive deadline exceeded")
        conn.settimeout(remaining)
        part = conn.recv(min(size - len(data), 65536))
        if not part:
            raise EOFError("Connection closed before complete message")
        data.extend(part)
    return bytes(data)


def recv_json(conn, deadline):
    size = struct.unpack("!I", recv_exact(conn, 4, deadline))[0]
    if not 0 < size <= MAX_JSON:
        raise ValueError("JSON length outside protocol limit")
    message = json.loads(recv_exact(conn, size, deadline))
    if not isinstance(message, dict):
        raise ValueError("Expected JSON object")
    return message


def send_json(conn, message, timeout):
    data = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_JSON:
        raise ValueError("Reply exceeds protocol limit")
    conn.settimeout(timeout)
    conn.sendall(struct.pack("!I", len(data)) + data)


def check_hello(hello, profile):
    if hello.get("type") != "HELLO" or integer(hello.get("version"), "version", 1, 1) != 1:
        raise ValueError("Expected HELLO v1")
    run_id = hello.get("run_id")
    if not isinstance(run_id, str) or not 0 < len(run_id) <= 256:
        raise ValueError("Invalid run_id")
    for key in ("width", "height"):
        if integer(hello.get(key), key, 1, MAX_IMAGE) != profile[key]:
            raise ValueError("Profile mismatch: " + key)
    for key in ("calibration_id", "distortion", "pixel_convention"):
        if hello.get(key) != profile[key]:
            raise ValueError("Profile mismatch: " + key)
    if hello.get("pixel_format") != "GRAY8":
        raise ValueError("Expected GRAY8")
    for key in ("focal_px", "cx", "cy"):
        if not math.isclose(finite(hello.get(key), key), profile[key], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("Profile mismatch: " + key)


def handle_connection(conn, tracker, identity, timeout=10.):
    """One frame per connection, exactly matching the current C++ host."""
    hello = recv_json(conn, time.monotonic() + timeout)
    check_hello(hello, tracker.profile)
    p = tracker.profile
    send_json(conn, dict(type="READY", version=1, width=p["width"], height=p["height"],
                        pixel_format="GRAY8", calibration_id=p["calibration_id"], **identity), timeout)
    deadline = time.monotonic() + timeout
    header = recv_json(conn, deadline)
    if header.get("type") != "IMAGE" or integer(header.get("version"), "version", 1, 1) != 1:
        raise ValueError("Expected IMAGE v1")
    if header.get("run_id") != hello["run_id"]:
        raise ValueError("IMAGE run_id mismatch")
    frame_id = integer(header.get("frame_id"), "frame_id", 0, 2147483647)
    for key in ("width", "height"):
        if integer(header.get(key), key, 1, MAX_IMAGE) != p[key]:
            raise ValueError("IMAGE dimensions mismatch")
    size = integer(header.get("payload_bytes"), "payload_bytes", 1, MAX_IMAGE)
    if size != p["width"] * p["height"] or header.get("pixel_format") != "GRAY8":
        raise ValueError("Invalid IMAGE size/format")
    checksum = header.get("image_crc32")
    if not isinstance(checksum, str) or len(checksum) != 8 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid CRC32 field")
    raw = recv_exact(conn, size, deadline)
    actual_crc = "{:08x}".format(zlib.crc32(raw) & 0xffffffff)
    result = dict(type="RESULT", version=1, run_id=header["run_id"],
                  frame_id=frame_id, image_crc32=actual_crc)
    if checksum != actual_crc:
        result.update(status="input_error", reason="image_crc32_mismatch")
    else:
        start = time.perf_counter()
        try:
            image = np.frombuffer(raw, dtype=np.uint8).reshape(p["height"], p["width"])
            result.update(tracker.process_one_frame(image))
        except Exception as exc:
            reset_matcher(tracker.matcher)
            result.update(status="input_error", reason=type(exc).__name__ + ": " + str(exc))
        result["processing_ms"] = (time.perf_counter() - start) * 1000.
    send_json(conn, result, timeout)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ultra96 CNN tracker PIL TCP server (copied main)")
    parser.add_argument("--profile", help="Local ideal-camera JSON profile")
    parser.add_argument("--xmodel", required=True, help="Actual deployed model path")
    parser.add_argument("--bitfile", default="dpu.bit")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind IPv4, preferably USB interface address")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=10.)
    parser.add_argument("--inspect-model", action="store_true", help="Load DPU and print actual tensor sizes, then exit")
    parser.add_argument("--log", default="pil_board_results.jsonl", help="Append result/connection diagnostics")
    args = parser.parse_args()
    integer(args.port, "port", 1, 65535)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if not args.inspect_model and not args.profile:
        parser.error("--profile required for server mode")
    # Resolve CLI paths before selecting the script directory for legacy catalog paths.
    xmodel, bitfile = Path(args.xmodel).resolve(), Path(args.bitfile).resolve()
    profile = load_profile(args.profile) if args.profile else None
    log_path = Path(args.log).resolve()
    os.chdir(SCRIPT_PATH.parent)
    nn_ctx = init_neural_net(str(xmodel), str(bitfile))
    model_info = {key: nn_ctx[key] for key in ("in_shape", "out_shape", "in_layout", "out_layout", "width", "height")}
    print(json.dumps(model_info), flush=True)
    if args.inspect_model:
        return
    if (profile["width"], profile["height"]) != (nn_ctx["width"], nn_ctx["height"]):
        raise ValueError("Profile dimensions do not match the loaded DPU model")
    from geometric_voting import geometric_voting
    tracker = Tracker(profile, lambda image: run_neural_net(image, nn_ctx), geometric_voting())
    identity = dict(device="Ultra96 CNN PIL", model_version=xmodel.name,
                    model_sha256=sha256_file(xmodel), input_layout=nn_ctx["in_layout"],
                    output_layout=nn_ctx["out_layout"],
                    catalog_sha256=sha256_file("catalog_data/star_catalog_6.1.csv"),
                    star_pairs_sha256=sha256_file("catalog_data/star_pair_6.1.csv"),
                    software_sha256=sha256_file(SCRIPT_PATH))
    with log_path.open("a", encoding="utf-8") as log, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen(4)
        print("PIL listening on {}:{}; DPU initialized once".format(args.bind, args.port), flush=True)
        try:
            while True:
                conn, peer = listener.accept()
                with conn:
                    try:
                        record = handle_connection(conn, tracker, identity, args.timeout)
                    except (ValueError, OSError, EOFError) as exc:
                        record = dict(event="connection_error", reason=str(exc))
                        try:
                            send_json(conn, dict(type="ERROR", version=1, reason=str(exc)), args.timeout)
                        except OSError:
                            pass
                record["peer"] = peer[0]
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(record.get("status", record.get("event")), record.get("frame_id", ""), flush=True)
        except KeyboardInterrupt:
            print("PIL server stopped", flush=True)


if __name__ == "__main__":
    main()
