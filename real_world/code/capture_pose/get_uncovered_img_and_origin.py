import cv2
import cv2.aruco as aruco
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from datetime import date
from sympy import Point, Line
import matplotlib.pyplot as plt
import argparse
import os.path as osp
import tqdm


    
def construct_sim_origin(center_coords, dtype=float):
    print(center_coords)
    x1 = center_coords[0][0]
    y1 = center_coords[0][1]
    x2 = center_coords[2][0]
    y2 = center_coords[2][1]

    x3 = center_coords[1][0]
    y3 = center_coords[1][1]
    x4 = center_coords[3][0]
    y4 = center_coords[3][1]

    line1 = Line(Point(x1, y1), Point(x2, y2))
    line2 = Line(Point(x3, y3), Point(x4, y4))
    intersection = line1.intersection(line2)[0]
    
    return np.array(intersection, dtype=dtype)

#https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
def show_aruco_tags(img, dist, mtx):
    image = img.copy()
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)
    #distance detection
    markerSizeInCM = 5
    rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
    centers = {}
    metercenters = {id:i[0]/100 for ([id],i) in zip(ids,tvec)}
    # print(metercenters)
    #draw on image
    if len(corners)>0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            centers[markerID]=[cX,cY]
            
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(image, str(markerID),(int(topLeft[0]), int(topLeft[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, centers, metercenters
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='TEST')
    args = parser.parse_args()
    # today = date.today()

    #start realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('141722070195')
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
    pipeline.start(config)

    for i in tqdm.tqdm(range(30)):
        pipeline.wait_for_frames()
    
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # #calculating camera intrinsics
    intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
    dist = np.array(intrinsics.coeffs)
    mtx = np.array([[intrinsics.fx, 0, intrinsics.width],[0,intrinsics.fy,intrinsics.height],[0,0,1]])

    pipeline.stop()

    img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    filename = 'uncovered_rgb.png'
    cv2.imwrite(osp.join(args.save_dir, filename), img)
    print(f'saved: {filename}')

    image = cv2.imread(osp.join(args.save_dir, filename))   
    aru, centers, metercenters = show_aruco_tags(image, dist, mtx)
    origin_px = construct_sim_origin(centers, int)
    origin_m = construct_sim_origin(metercenters)
    m2px_scale = origin_m/origin_px             # this is the ratio of meters to pixels

    with open(osp.join(args.save_dir,'sim_origin_data.pkl'), 'wb') as f:
        pickle.dump({
            'dist':dist, 
            'mtx':mtx,
            'centers_px':centers,
            'centers_m':metercenters,
            'origin_px':origin_px,
            'origin_m':origin_m,
            'm2px_scale':m2px_scale
            }, f)