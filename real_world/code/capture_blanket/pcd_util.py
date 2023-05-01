
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
import cv2

def get_pointcloud(pipeline):
    pc = rs.pointcloud()
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    intrinsics_depth = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()  
    color_frame = aligned_frames.get_color_frame()
    intrinsics_color = color_frame.profile.as_video_stream_profile().get_intrinsics()  
    color_image = np.array(color_frame.get_data())

    # viz_rgb_as_hsv(color_image)
    # viz_color_segmention(color_image, mask_params=[0.45, 0.57, 0.45])

    pc.map_to(color_frame)
    # Generate the pointcloud and texture mappings
    points = pc.calculate(aligned_depth_frame)
    points.export_to_ply("temp.ply", color_frame)

    pcd = o3d.io.read_point_cloud("temp.ply")

    return pcd, color_image, (intrinsics_depth, intrinsics_color)

def process_and_save_pcd(pcd, color_image, mask_params, z_thresholds=None, y_threshold=None, x_threshold=None, intrinsics = None, name=None, save_dir=None, subject_dir=None):
    # pcd = o3d.io.read_point_cloud("temp.ply")
    filt_pcd = filter_pcd(pcd, mask_params, z_thresholds, y_threshold, x_threshold)
    if intrinsics is not None:
        if subject_dir is not None:
            filt_pcd = transform_to_sim_origin(filt_pcd, intrinsics, subject_dir)
        else:
            filt_pcd = transform_to_sim_origin(filt_pcd, intrinsics, save_dir)
    cv2.imwrite(osp.join(save_dir, f'covered_rgb_{name}.png'), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    filename = f'pcd_filtered_{name}.pcd'
    o3d.io.write_point_cloud(osp.join(save_dir, filename), filt_pcd)
    print(f"Saving to {filename}...")
    print("Done")

    return filt_pcd


def filter_pcd(pcd, mask_params, z_thresholds, y_threshold, x_threshold):
    '''
    see the following for information on color image segmentation:
    https://mattmaulion.medium.com/color-image-segmentation-image-processing-4a04eca25c0
    '''

    colors_rgb = np.asarray(pcd.colors)

    colors_hsv = np.array([colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]) for rgb in colors_rgb])

    lower_mask = colors_hsv[:,0] > mask_params[0]       #* refer to hue channel (value from the colorbar)
    upper_mask = colors_hsv[:,0] < mask_params[1]       #* refer to hue channel (value from the colorbar)
    saturation_mask = colors_hsv[:,1] > mask_params[2]  #* refer to transparency channel (value from the colorbar)
    
    mask = upper_mask*lower_mask*saturation_mask

    #* apply segmentation mask to pcd
    pcd_color_seg = pcd.select_by_index(np.where(mask)[0])

    # return pcd_color_seg

    #* filter out all points
    points = np.asarray(pcd_color_seg.points)

    thres_points = np.ones(len(points)) != 0
    if z_thresholds is not None:
        z_min, z_max = z_thresholds
        thres_points = ((points[:, 2] <= -z_min) & (points[:, 2] > -z_max))
    if y_threshold is not None:
        # thres_points = thres_points & (points[:, 1] > -y_threshold)
        thres_points = thres_points & (points[:, 1] > y_threshold)
    if x_threshold is not None:
        
        thres_points = thres_points & (points[:, 0] > x_threshold)
    pcd_filt = pcd_color_seg.select_by_index(np.where(thres_points)[0])

    return pcd_filt

def transform_to_sim_origin(pcd, intrinsics, save_dir):
    pcd_origin_px = np.array(rs.rs2_project_point_to_pixel(intrinsics[0], [0.0, 0.0, 1.3]), dtype=int)

    with open(osp.join(save_dir, 'sim_origin_data.pkl'),'rb') as f:
        data = pickle.load(f)
        dist = data['dist']
        mtx = data['mtx']
        centers_px = data['centers_px']
        centers_m = data['centers_m']
        origin_px = data['origin_px']
        origin_m = data['origin_m']
        m2px_scale = data['m2px_scale']
    trans = np.append((pcd_origin_px - origin_px)*m2px_scale, 0)
    return pcd.translate(trans)

def viz_rgb_as_hsv(rgb_img):
    ''' 
    visualizes an RGB image as an HSV image
    run this function to find mask parameters to feed into color segmentation
    '''

    print(rgb_img.shape)
    hsv_img = rgb2hsv(rgb_img)
    fig, ax = plt.subplots(2, 3, figsize=(12,8))
    ax[0][0].imshow(hsv_img[:,:,0], cmap='gray')
    ax[0][0].set_title('Hue')
    ax[0][1].imshow(hsv_img[:,:,1], cmap='gray')
    ax[0][1].set_title('Saturation')
    ax[0][2].imshow(hsv_img[:,:,2], cmap='gray')
    ax[0][2].set_title('Value')
    # plt.show()

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[1][0].imshow(hsv_img[:,:,0],cmap='hsv')
    ax[1][0].set_title('hue')
    ax[1][1].imshow(hsv_img[:,:,1],cmap='hsv')
    ax[1][1].set_title('transparency')
    ax[1][2].imshow(hsv_img[:,:,2],cmap='hsv')
    ax[1][2].set_title('value')
    fig.colorbar(imshow(hsv_img[:,:,0],cmap='hsv')) 
    fig.tight_layout()
    fig.savefig('color_img')
    plt.show(block=True)
    

def viz_color_segmention(rgb_img, mask_params):
    ''' 
    visualizes an RGB image after application of a segmentation mask
    run this function to verify mask parameters used for color segmentation
    '''

    hsv_img = rgb2hsv(rgb_img)

    lower_mask = hsv_img[:,:,0] > mask_params[0]        #* refer to hue channel (values from the colorbar)
    upper_mask = hsv_img[:,:,0] < mask_params[1]        #* refer to hue channel (values from the colorbar)
    saturation_mask = hsv_img[:,:,1] > mask_params[2]   #* refer to transparency channel (values from the colorbar)
    
    mask = upper_mask*lower_mask*saturation_mask
    print(mask)
    # mask = upper_mask*lower_mask
    red = rgb_img[:,:,0]*mask
    green = rgb_img[:,:,1]*mask
    blue = rgb_img[:,:,2]*mask
    masked_img = np.dstack((red,green,blue))

    plt.figure()
    imshow(masked_img)
    plt.show()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, axis])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, data_dir, left=True):
    print(":: Load two point clouds and disturb initial pose.")

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # Load the point clouds
    target = o3d.io.read_point_cloud(osp.join(data_dir, "pcd_filtered_top.pcd"))

    if left:
        source = o3d.io.read_point_cloud(osp.join(data_dir,"pcd_filtered_left.pcd"))
    else:
        source = o3d.io.read_point_cloud(osp.join(data_dir,"pcd_filtered_right.pcd"))

    if left:
        trans_init = np.asarray([[ 0.74640198, -0.31413591,  0.58668792,  0.8653032 ],
                                [ 0.6607477,   0.45492902, -0.59703607, -0.91064895],
                                [-0.07935089,  0.8332816,   0.5471245,  -1.10805079],
                                [ 0.,          0.,          0.,          1.        ]])
    else:
        trans_init = np.asarray([[-0.99847948, -0.03957142,  0.03837784,  0.00775568],
                                [ 0.05153048, -0.42274173,  0.90478394,  2.05092663],
                                [-0.01957968,  0.90538586,  0.42413801, -1.1368759 ],
                                [ 0.,          0.,          0.,          1.        ]])

    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


# OLD VERSION
# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.3
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999))
#     return result
