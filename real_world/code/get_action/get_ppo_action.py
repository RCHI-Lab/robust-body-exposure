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
from assistive_gym.learn import evaluate_policy_real_world
from build_bm_graph import BM_Graph
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *
from gnn_train_test_new import GNN_Train_Test
from gym.utils import seeding
from tqdm import tqdm
import open3d as o3d
import argparse

def remap_action_ppo(action, remap_ranges):
    remap_action = []
    for i in range(len(action)):
        a = np.interp(action[i], [-1, 1], remap_ranges[i])
        remap_action.append(a)
    return np.array(remap_action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', type=str, default='TEST')
    parser.add_argument('--pose-dir', type=str, default='TEST')
    parser.add_argument('--tl-code', type=str)
    parser.add_argument('--manikin', type=int, default=0)
    args = parser.parse_args()

    policy_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/PPO_rw'
    # target_limb_code = 13
    target_limb_code = int(args.tl_code)
    # print(f'      TARGET LIMB CODE: {target_limb_code}')

    pose_dir = args.pose_dir
    if args.manikin:
        pose_dir = args.subject_dir
    with open(osp.join(pose_dir,'human_pose.pkl'), 'rb') as f:
        human_m_correct_axis = pickle.load(f)


    observation = human_m_correct_axis.flatten()
    action, elapsed_time = evaluate_policy_real_world(policy_path, target_limb_code, observation)
    # action = remap_action_ppo(action, remap_ranges=[[0, 1], [0.5, 1], [0, 1], [1, 1]])
    print(action)
    action = remap_action_ppo(action, remap_ranges=[[0, 1], [-0.5, 1], [0, 1], [-1, 1]])
    print(action)
    scaled_action = scale_action(action)

    with open('scaled_action.pkl', 'wb') as f:
        pickle.dump(scaled_action, f, protocol=2)
    
    print(scaled_action)


    pcd = o3d.io.read_point_cloud(osp.join(args.pose_dir,'blanket_pcd.pcd'))
    points = np.asarray(pcd.points)
    thres_points = ((points[:, 1] <= 0.6) & (points[:, 1] >= -0.6) & (points[:, 0] <= 0.8) & (points[:, 0] > -1) & (points[:, 2] > -1.8))
    pcd_filt = pcd.select_by_index(np.where(thres_points)[0])
    points = np.asarray(pcd_filt.points)
    R = get_rotation_matrix([0,0,1], np.pi/2)
    points = points @ R
    points[:, 1] += 0.1  # translation correction

    cloth_initial_raw = points

    dist, is_on_cloth = check_grasp_on_cloth(scaled_action, np.array(cloth_initial_raw))
    print('on cloth? :', is_on_cloth)

    # with open(osp.join(args.subject_dir,'body_info.pkl'),'rb') as f:
    #     body_info = pickle.load(f)

    # all_body_points = get_body_points_from_obs(human_pose=human_m_correct_axis.astype(float), target_limb_code=target_limb_code, body_info=body_info)

    # # scaled_action = scale_action(best_action, x_range=[-0.44, 0])
    # cloth_initial = cloth_initial_raw[:,:2]
    # initial_covered_status = get_covered_status(all_body_points, cloth_initial)
    # fscore = compute_fscore(initial_covered_status, initial_covered_status)
    # print(scaled_action, 0, fscore)

    # fig = generate_figure(
    #         target_limb_code, 
    #         scaled_action, 
    #         body_info, 
    #         all_body_points, 
    #         cloth_initial = cloth_initial, 
    #         final_cloths = [cloth_initial],
    #         initial_covered_status = initial_covered_status,
    #         covered_statuses = [initial_covered_status] , 
    #         fscores = [fscore],
    #         plot_initial=True, compare_subplots=False,
    #         draw_axes = True)
    # # fig.write_image(osp.join(args.pose_dir, f'eval_figure_{fscore:.3f}.png'))
    # fig.write_image(osp.join(args.pose_dir, f'eval_figure.png'))
    # fig.show()

    # eval_data = {
    #     'action': action,
    #     'scaled_action': scaled_action,
    #     'cloth_initial': cloth_initial,
    #     'cloth_final': None,
    #     'reward': None,
    #     'fscore': None,
    #     'all_body_points':None,
    #     'body_info': body_info,
    #     'initial_covered_status': initial_covered_status,
    #     'final_covered_status':None,
    #     'target_limb_code':target_limb_code,
    #     'is_on_cloth':is_on_cloth,
    #     'best_time': elapsed_time,
    #     'best_fevals': None,
    #     'best_iterations': None
    #     }

    # with open(osp.join(args.pose_dir, 'cma_eval_data.pkl'), 'wb') as f:
    #     pickle.dump(eval_data, f)

