import numpy as np
#from camera_software import camera_software
import cv2
from star_detection_centroiding import *
from itertools import combinations

import torch 
from torchvision.transforms import Normalize
from scipy.linalg import lstsq

import time 
import sys

import argparse

from geometric_voting import geometric_voting
import math

import pandas as pd
from tqdm import trange
import os 

from pynq_dpu import DpuOverlay

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
                        if seg_map[row, col] > 0:
                            u = (col + 0.5) 
                            v = (row + 0.5) 
                            r_i.append(dist_map[row, col])
                            x_i.append(u)
                            y_i.append(v)
                            seg_map[row, col] = 0 # avoid getting reused
                    except IndexError:
                        pass
            n = len(x_i)
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
            u = x[0][0] * pixel_size
            v = x[0][1] * pixel_size
            centroid_est.append([u, v])

            if len(centroid_est) > 50: # too many detection
                return centroid_est  

    return centroid_est

def run_neural_net(img):
    
    pixel_size = 1 # get centroid in pixel instead of mm
    radius = 7

    # for data_Oct19
    mean = [25.36114133]
    std =  [44.31162568]

    overlay = DpuOverlay("dpu.bit")              # 你的 overlay 名稱
    overlay.load_model("./CNN_Star_Tracker_Model_V1.xmodel")      # 你編好的 xmodel
    runner = overlay.runner

    #取得 tensor 屬性
    inp_t = runner.get_input_tensors()[0]
    out_t = runner.get_output_tensors()[0]
    in_shape = inp_t.dims # e.g. [1,1,480,640]
    in_fix = inp_t.get_attr("fix_point")
    out_fix = out_t.get_attr("fix_point")

    # img_np: (H,W) float32/float64
    # 前處理：normalize + 量化成 int8
    img_f = img.astype(np.float32)
    img_f = (img_f - mean) / std
    scale = 2 ** IN_FIX
    inp_int8 = np.round(img_f * scale).astype(np.int8).reshape(IN_SHAPE)

    # 配置 buffer
    input_buf = [np.empty_like(inp_int8)]
    output_buf = [np.empty(out_t.dims, dtype=np.int8)]
    input_buf[0][...] = inp_int8

    # 執行
    jid = runner.execute_async(input_buf, output_buf)
    runner.wait(jid)

    # 反量化輸出
    out_scale = 2 ** OUT_FIX
    out_f = output_buf[0].astype(np.float32) / out_scale  # shape: [1,2,H,W]
    seg_prediction = (1.0 / (1.0 + np.exp(-out_f[0,0]))) > 0.5       # segmentation mask
    dist_prediction = out_f[0,1]                                     # distance map

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

    print(f'running {args.mode}')

    animation = []

    for i in trange(video.shape[0], smoothing=0):
        
        img = video[i].astype('float32')

        if args.mode == 'NN':
            centroid_est = run_neural_net(img.copy())
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

        cv2.imshow('Star Detection and Centroiding', img) 
        cv2.imshow('Angular Distacne', vis) 
        animation.append(img)

        print('Num of Centroids = {}, num of ids = {}, att=[{:.4f},{:.4f},{:.4f}]'.format(len(centroid_est), len(geometric_voting_obj.star_vectors_ver), phi, theta, psi))
        # two lines below is to erase the previous print message
        sys.stdout.write("\033[F") # Cursor up one line
        sys.stdout.write("\033[K") # Clear to the end of line ( if you print something shorter than before)

        k = cv2.waitKey(1)
        if k == ord('q'):
            print("Close Program")
            break

    print("save star detection and centroiding results")
    np.save(f'./saved_results/attitude_{args.mode}_{i}.npy', np.asarray(attitude))
    np.save(f'./saved_results/id_rate_{args.mode}_{i}.npy', np.asarray(id_rate))
    np.save(f'./saved_results/animation_{args.mode}_{i}.npy', np.asarray(animation))

if __name__ == '__main__':
    """
        python .\main_detection_centroiding.py --mode NN --input video --video_file video_Test3.npy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='baseline')
    parser.add_argument("--save_video", action='store_true')
    parser.add_argument("--input", type=str, default='camera')
    parser.add_argument("--video_file", type=str, default=None)


    args = parser.parse_args() 

    if args.input == 'camera':
        # main_camera(args)
        print('We don't support camera input now')
    elif args.input == 'video':
        main_video(args)
    else:
        print('Wrong Input Type')