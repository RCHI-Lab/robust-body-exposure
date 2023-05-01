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
from pcd_util import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--subject-dir', type=str, default='TEST')
parser.add_argument('--pose-dir', type=str, default='TEST')
parser.add_argument('--manikin', type=int, default=0)
args = parser.parse_args()

w = 1280
h = 720
fps = 6

camera_ids = ['141722070195', '817612070529', '818312070704'] # top, right, and left
camera_names = ['top', 'right', 'left']


pipeline_top = rs.pipeline()
config_top = rs.config()
config_top.enable_device(camera_ids[0]) # top camera
config_top.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config_top.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)

pipeline_right = rs.pipeline()
config_right = rs.config()
config_right.enable_device(camera_ids[1]) # top camera
config_right.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config_right.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)

pipeline_left = rs.pipeline()
config_left = rs.config()
config_left.enable_device(camera_ids[2]) # top camera
config_left.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
config_left.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)


# Start streaming
pipeline_top.start(config_top)
pipeline_right.start(config_right)
pipeline_left.start(config_left)

# Processing blocks

for i in tqdm(range(30)):
    pipeline_top.wait_for_frames()
    pipeline_right.wait_for_frames()
    pipeline_left.wait_for_frames()

pcd_top, rgb_top, intrinsics = get_pointcloud(pipeline_top)
pcd_right, rgb_right, _ = get_pointcloud(pipeline_right)
pcd_left, rgb_left, _ = get_pointcloud(pipeline_left)

mask_params = [0.45, 0.57, 0.45]

if args.manikin:
    pcd_top = process_and_save_pcd(pcd_top, rgb_top, mask_params, z_thresholds= (1.3, 1.6), intrinsics = intrinsics, name='top', save_dir = args.pose_dir, subject_dir=args.subject_dir)
    pcd_right = process_and_save_pcd(pcd_right, rgb_right, mask_params, z_thresholds=(0, 2.2), name='right', save_dir = args.pose_dir, subject_dir=args.subject_dir)
    pcd_left = process_and_save_pcd(pcd_left, rgb_left, mask_params, z_thresholds=(0, 2), name='left', save_dir = args.pose_dir, subject_dir=args.subject_dir)
else:
    pcd_top = process_and_save_pcd(pcd_top, rgb_top, mask_params, z_thresholds= (1.3, 1.6), intrinsics = intrinsics, name='top', save_dir = args.pose_dir)
    pcd_right = process_and_save_pcd(pcd_right, rgb_right, mask_params, z_thresholds=(0, 2.2), name='right', save_dir = args.pose_dir)
    pcd_left = process_and_save_pcd(pcd_left, rgb_left, mask_params, z_thresholds=(0, 2), name='left', save_dir = args.pose_dir)

## o3d.visualization.draw_geometries([pcd_top])
# o3d.visualization.draw_geometries([pcd_right])
# o3d.visualization.draw_geometries([pcd_left])

# pcd = o3d.io.read_point_cloud("1.ply")
# o3d.visualization.draw_geometries([pcd])

pipeline_top.stop()
pipeline_right.stop()
pipeline_left.stop()
# %%

print("Done capturing")
print("Starting registration")


voxel_size = 0.0225  # means 2.25cm for the dataset
# voxel_size = 0.025  # means 2.5cm for the dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, args.pose_dir)

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
print("Transformation is:")
print(result_ransac.transformation)
draw_registration_result(source_down, target_down, result_ransac.transformation)

# apply the transformation to the original point cloud
transform_left = result_ransac.transformation
source.transform(transform_left)
left_pcd = source

source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, args.pose_dir, left=False)

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

print(result_ransac)
print("Transformation is:")
print(result_ransac.transformation)
draw_registration_result(source_down, target_down, result_ransac.transformation)

# apply the transformation to the right point cloud
transform_right = result_ransac.transformation
source.transform(transform_right)
right_pcd = source

# save the transformation matrices
np.save(osp.join(args.pose_dir, "transform_left.npy"), transform_left)
np.save(osp.join(args.pose_dir, "transform_right.npy"), transform_right)

# save the point clouds
o3d.io.write_point_cloud(osp.join(args.pose_dir, "left_pcd.pcd"), left_pcd)
o3d.io.write_point_cloud(osp.join(args.pose_dir, "right_pcd.pcd"), right_pcd)


# merge the three point clouds
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate((left_pcd.points, right_pcd.points, target.points), axis=0))
merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate((left_pcd.colors, right_pcd.colors, target.colors), axis=0))

# visualize the merged point cloud
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([merged_pcd, axis])

# save the merged point cloud
o3d.io.write_point_cloud(osp.join(args.pose_dir,"blanket_pcd.pcd"), merged_pcd)
