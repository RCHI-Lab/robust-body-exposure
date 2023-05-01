#%%
#%%
# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

import copy
import pickle
import os
import os.path as osp
import argparse
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import open3d as o3d
import colorsys
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default='TEST')
args = parser.parse_args()

w = 1280
h = 720
fps = 6

camera_ids = ['141722070195', '817612070529', '818312070704'] # top, right, and left
camera_names = ['top', 'right', 'left']


pipeline = rs.pipeline()
config = rs.config()
config.enable_device(camera_ids[int(args.camera)]) # top camera
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)


# Start streaming
pipeline.start(config)

# # Processing blocks
# for i in tqdm(range(30)):
#     pipeline.wait_for_frames()

while True:
    pc = rs.pointcloud()
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    intrinsics_depth = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()  
    color_frame = aligned_frames.get_color_frame()
    intrinsics_color = color_frame.profile.as_video_stream_profile().get_intrinsics()  
    color_image = np.array(color_frame.get_data())

    cv2.imshow("Image",color_image)
    if cv2.waitKey(1) == ord("q"):
        break


pipeline.stop()