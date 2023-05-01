#%%
import math
import sys
import time

# import multiprocessing
from torch import multiprocessing

sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns')
import os.path as osp
from pathlib import Path

import cma
import numpy as np
import torch
from assistive_gym.envs.bu_gnn_util import *
from assistive_gym.learn import make_env
from build_bm_graph import BM_Graph
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *
from gnn_train_test_new import GNN_Train_Test
from gym.utils import seeding
from tqdm import tqdm
import open3d as o3d
import argparse

def get_naive_action(human_pose, target_limb_code, data):
    # target_limb_code = 8
    all_limb_endpoints = [
        [0,1], [0,1], [0,2],
        [3,4], [3,4], [3,5],
        [6,7], [6,7], [6,8],
        [9,10], [9,10], [9,11],
        [3,9,4,10], [2,8,5,11], [3,9,5,11],
        [2,8,3,9]
    ]
    limb_endpoints = all_limb_endpoints[target_limb_code]

    if len(limb_endpoints) == 2:
        p1 = human_pose[limb_endpoints[0]]
        p2 = human_pose[limb_endpoints[1]]
    elif len(limb_endpoints) == 4:
        p1 = (human_pose[limb_endpoints[0]] + human_pose[limb_endpoints[1]])/2
        p2 = (human_pose[limb_endpoints[2]] + human_pose[limb_endpoints[3]])/2
                                                    
                                                        # excluding single points (hands and feet)
    midpoint = (p1 + p2)/2 if target_limb_code not in [0,3,6,9] else human_pose[limb_endpoints[0]]
    # print(p1, p2, midpoint)
    dists = []
    points = []

    axis_vector = p1-p2
    
    # points on legs or combo targets
    if target_limb_code in [3,4,5,9,10,11,12,13,14,15]:      # grasp is colinear with axis vector
        trajectory_vector = axis_vector
    # left arm vs right arm
    else:                                                    # grasp is normal to axis vector, for hands, forearms, arms only
        direction = [-1, 1] if target_limb_code in [6,7,8] else [1, -1]
        trajectory_vector = np.array([axis_vector[1], axis_vector[0]])*np.array(direction)

    if target_limb_code == 15: # to uncover the whole body, pull the blanket to the foot of the bed
        r_y = 1.05
        r_x = r_y*trajectory_vector[0]/trajectory_vector[1]
        release = [r_x, r_y]
    else:
        for i, p3 in enumerate(data):
            cloth_vector = p3[0:2]-midpoint
            # cloth_vector = midpoint-p3[0:2] if target_limb_code in [3,4,5,9,10,11] else p3[0:2]-midpoint

            # deviation of a given point on the cloth and the trajectory direction
            d = np.linalg.norm(np.cross(trajectory_vector, cloth_vector))/np.linalg.norm(trajectory_vector)

            trajectory_vector = trajectory_vector/np.linalg.norm(trajectory_vector)
            cloth_vector = cloth_vector/np.linalg.norm(cloth_vector)
            theta = np.arccos(np.clip(np.dot(trajectory_vector, cloth_vector), -1.0, 1.0))
            # if d < 0.01 and p3[2]>=0.58 and theta < np.pi/2:
            if d < 0.04 and p3[2]>=-1.6 and theta < np.pi/2:
                # print(theta)
                dists.append(np.linalg.norm(midpoint-p3[0:2]))
                points.append(p3)
                # self.create_sphere(radius=0.01, mass=0.0, pos = p3, visual=True, collision=True, rgba=[1, 0, 0, 1])
        
        print(trajectory_vector)
        #! ADD CONDITION FOR IF NO CLOTH POINTS FOUND, maybe increase d thers?
        if len(dists) == 0:
            return None
        release = midpoint + (midpoint - points[np.argmax(dists)][0:2])
        # p.addUserDebugText(text=str('edge'), textPosition=points[np.argmax(dists)], textColorRGB=[0, 0, 0], textSize=1, lifeTime=0, physicsClientId=self.id)

    grasp = p1 if target_limb_code in [4, 5, 10, 11, 12, 13, 14, 15] else list(points[np.argmax(dists)][0:2])

    # constrain to the bed (sometimes limbs or cloth is slightly outside these bounds)
    grasp[0] = np.clip(grasp[0], -0.44, 0.44)
    grasp[1] = np.clip(grasp[1], -1.05, 1.05)
    release[0] = np.clip(release[0], -0.44, 0.44)
    release[1] = np.clip(release[1], -1.05, 1.05)

    action = np.array([grasp[0], grasp[1], release[0], release[1]])
    
    #! ClIP HERE
    action[0], action[2] = list(np.clip([action[0], action[2]], -0.44, 0.44))
    action[1], action[3] = list(np.clip([action[1], action[3]], -1.05, 1.05))

    return action

def sub_sample_point_clouds(cloth_initial_3D_pos):
    cloth_initial = np.array(cloth_initial_3D_pos)

    voxel_size = 0.05
    # nb_vox=np.ceil((np.max(cloth_initial, axis=0) - np.min(cloth_initial, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((cloth_initial - np.min(cloth_initial, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)

    voxel_grid={}
    voxel_grid_cloth_inds={}
    cloth_initial_subsample=[]
    last_seen=0
    for idx,vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)]= cloth_initial[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        voxel_grid_cloth_inds[tuple(vox)] = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
        
        closest_point_to_barycenter = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
        cloth_initial_subsample.append(voxel_grid[tuple(vox)][closest_point_to_barycenter])

        last_seen+=nb_pts_per_voxel[idx]

    return cloth_initial_subsample


def crop_action(action):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', type=str, default='TEST')
    parser.add_argument('--pose-dir', type=str, default='TEST')
    parser.add_argument('--tl-code', type=str)
    parser.add_argument('--manikin', type=int, default=0)
    args = parser.parse_args()


    pcd = o3d.io.read_point_cloud(osp.join(args.pose_dir,'blanket_pcd.pcd'))
    points = np.asarray(pcd.points)
    thres_points = ((points[:, 1] <= 0.6) & (points[:, 1] >= -0.6) & (points[:, 0] <= 0.8) & (points[:, 0] > -1) & (points[:, 2] > -1.8))
    pcd_filt = pcd.select_by_index(np.where(thres_points)[0])
    points = np.asarray(pcd_filt.points)
    R = get_rotation_matrix([0,0,1], np.pi/2)
    points = points @ R
    points[:, 1] += 0.1  # translation correction
    print(points)

    cloth_initial_raw = points

    cloth_subsample = np.array(sub_sample_point_clouds(cloth_initial_raw))
    # target_limb_code = 13
    target_limb_code = int(args.tl_code)
    # print(f'      TARGET LIMB CODE: {target_limb_code}')

    pose_dir = args.pose_dir
    if args.manikin:
        pose_dir = args.subject_dir
    with open(osp.join(pose_dir,'human_pose.pkl'), 'rb') as f:
        human_m_correct_axis = pickle.load(f)

    action = get_naive_action(human_m_correct_axis, target_limb_code, data=cloth_subsample)
    scaled_action = scale_action(action, scale=[1, 1])
    with open('scaled_action.pkl', 'wb') as f:
        pickle.dump(scaled_action, f, protocol=2)
    print(scaled_action)

    import matplotlib.pyplot as plt
    plt.scatter(cloth_subsample[:, 0], cloth_subsample[:, 1])
    plt.scatter(human_m_correct_axis[:, 0], human_m_correct_axis[:, 1])
    plt.scatter(scaled_action[0], scaled_action[1])
    plt.scatter(scaled_action[2], scaled_action[3])
    plt.show()

    # import matplotlib.pyplot as plt
    # plt.scatter(cloth_subsample[:, 0], cloth_subsample[:, 2])
    # # plt.scatter(human_m_correct_axis[:, 0], human_m_correct_axis[:, 1])
    # plt.scatter(scaled_action[0], -1.6)
    # plt.scatter(scaled_action[2], -1.6)
    # plt.show()

    dist, is_on_cloth = check_grasp_on_cloth(scaled_action, np.array(cloth_initial_raw))
    print('on cloth? :', is_on_cloth)


    # # scaled_action = scale_action(best_action, x_range=[-0.44, 0])
    # cloth_initial = graph.initial_blanket_state
    # initial_covered_status = get_covered_status(all_body_points, cloth_initial)
    # fscore = compute_fscore(initial_covered_status, best_covered_status)
    # print(scaled_action, best_reward, fscore)

    # fig = generate_figure(
    #         target_limb_code, 
    #         scaled_action, 
    #         body_info, 
    #         all_body_points, 
    #         cloth_initial = cloth_initial, 
    #         final_cloths = [best_pred],
    #         initial_covered_status = cloth_initial,  # why did i do this initial lol?
    #         initial_covered_status = initial_covered_status,
    #         covered_statuses = [best_covered_status] , 
    #         fscores = [fscore],
    #         plot_initial=True, compare_subplots=False,
    #         draw_axes = True)
    # # fig.write_image(osp.join(args.pose_dir, f'eval_figure_{fscore:.3f}.png'))
    # fig.write_image(osp.join(args.pose_dir, f'eval_figure.png'))
    # fig.show()

    # eval_data = {
    #     'action': best_action,
    #     'scaled_action': scaled_action,
    #     'cloth_initial': cloth_initial,
    #     'cloth_final': best_pred,
    #     'reward': best_reward,
    #     'fscore': fscore,
    #     'all_body_points':all_body_points,
    #     'body_info': body_info,
    #     'initial_covered_status': initial_covered_status,
    #     'final_covered_status':best_covered_status,
    #     'target_limb_code':target_limb_code,
    #     'is_on_cloth':best_is_on_cloth,
    #     'best_time':best_time,
    #     'best_fevals':best_fevals,
    #     'best_iterations':best_iterations
    #     }

    # with open(osp.join(args.pose_dir, 'cma_eval_data.pkl'), 'wb') as f:
    #     pickle.dump(eval_data, f)
    

