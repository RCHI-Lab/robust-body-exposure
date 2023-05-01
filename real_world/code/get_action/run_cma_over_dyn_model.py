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


# check if grasp is on the cloth BEFORE subsampling! cloth_initial_raw is pre subsampling
# ! maybe there is already a util function for this? check_grasp_on_cloth
def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D):
    action = scale_action(action)
    # action = scale_action(action,  x_range=[-0.44, 0])
    cloth_initial = graph.initial_blanket_state
    # if not grasp_on_cloth(action, cloth_initial_raw):
    #     return [0, cloth_initial, -1, None]
    is_on_cloth= grasp_on_cloth(action, cloth_initial_raw)
    if is_on_cloth:
        data = graph.build_graph(action)

        data = data.to(device).to_dict()
        batch = data['batch']
        batch_num = np.max(batch.data.cpu().numpy()) + 1
        # batch_num = np.max(batch.data.detach().cpu().numpy()) + 1    # version used for gpu, not cpu only for this script
        global_size = 0
        global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32, device=device)
        data['u'] = global_vec
        pred = model(data)['target'].detach().numpy()

        if use_disp:
            pred = cloth_initial + pred
    
    else:
        pred = np.copy(cloth_initial)

    
    if use_3D:
        cloth_initial_2D = np.delete(cloth_initial, 2, axis = 1)
        pred_2D = np.delete(pred, 2, axis = 1)
        # print('predicted', pred[0:10])
        cost, covered_status = get_cost(action, all_body_points, cloth_initial_2D, pred_2D)
    else:
        cost, covered_status = get_cost(action, all_body_points, cloth_initial, pred)

    return [cost, pred, covered_status, is_on_cloth]

def get_cost(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
    cost = -reward
    return cost, covered_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', type=str, default='TEST')
    parser.add_argument('--pose-dir', type=str, default='TEST')
    parser.add_argument('--tl-code', type=str)
    parser.add_argument('--manikin', type=int, default=0)
    args = parser.parse_args()


    trained_models_dir = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS'  # combo var
    model = 'standard_2D_10k_epochs=250_batch=100_workers=4_1668718872'


    checkpoint = osp.join(trained_models_dir, model)
    device = 'cpu'
    gnn_train_test = GNN_Train_Test(device)
    gnn_train_test.load_model_from_checkpoint(checkpoint)
    gnn_train_test.model.to(torch.device('cpu'))
    gnn_train_test.model.share_memory()
    gnn_train_test.model.eval()

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

    graph = BM_Graph(
        root = None,
        description=None,
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        cloth_initial=cloth_initial_raw,
        filter_draping=False,
        rot_draping=True,
        use_3D=False)

    # target_limb_code = 13
    target_limb_code = int(args.tl_code)
    # print(f'      TARGET LIMB CODE: {target_limb_code}')

    pose_dir = args.pose_dir
    if args.manikin:
        pose_dir = args.subject_dir
    with open(osp.join(pose_dir,'human_pose.pkl'), 'rb') as f:
        human_m_correct_axis = pickle.load(f)
    with open(osp.join(args.subject_dir,'body_info.pkl'),'rb') as f:
        body_info = pickle.load(f)

    all_body_points = get_body_points_from_obs(human_pose=human_m_correct_axis.astype(float), target_limb_code=target_limb_code, body_info=body_info)

    pop_size = 8
    maxfevals = 150

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': pop_size, 'maxfevals': maxfevals, 'tolfun': 1e-11, 'tolflatfitness': 20, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'maxfevals': 500
    # bounds = np.array([1]*4)
    bounds_max = np.array([1, 1, 1, 1])
    bounds_min = -1 * np.array([0, 0.5, 0, 1])
    opts.set('bounds', [bounds_min, bounds_max])
    opts.set('CMA_stds', bounds_max)

    # need to fix so that we don't get out of bounds guesses
    x0 = set_x0_for_cmaes(target_limb_code)

    print(f'Initial Guess: {x0}')
    for i in range(len(x0)):
        x0[i] =bounds_min[i] if x0[i] < bounds_min[i] else x0[i]

    
    print(f'Initial Guess Adjusted: {x0}')
    # if target_limb_code is 4:
    #     x0 = np.array([0, -0.4,  0,   0])
    # if target_limb_code is 8:
    #     x0 = np.array([0, -0.4,  0,   0])
    # x0 = np.array([0, 0.5,  0,   0])

    # x0 = [0.5, 0.5, -0.5, -0.5]     # * grasp scaled is at [0.22, 0.525]
    # x0 = [0, 0, 0, -0.5]
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.2
    reward_threshold = 95

    total_fevals = 0

    fevals = 0
    iterations = 0
    t0 = time.time()

    # best_action = np.array([0, 0.8, 0, -1])

    # * initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    # print('Env and CMA-ES set up')

    # # * evaluate cost function in parallel over num_cpus/4 processes
    # with EvalParallel2(cost_function, number_of_processes=num_proc) as eval_all:
    # * continue running optimization until termination criteria is reached
    best_cost = None
    while not es.stop():
        iterations += 1
        fevals += pop_size
        total_fevals += pop_size
        
        actions = es.ask()
        output = [cost_function(x, all_body_points, cloth_initial_raw, graph, gnn_train_test.model, device, use_disp=True, use_3D=False) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_statuses = output[2]
        is_on_cloth = output[3]
        # print(-1*np.array(costs))
        es.tell(actions, costs)
        
        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_statuses[best_cost_ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
            best_is_on_cloth = is_on_cloth[best_cost_ind]
        if best_reward >= reward_threshold:
            break
    scaled_action = scale_action(best_action)

    with open('scaled_action.pkl', 'wb') as f:
        pickle.dump(scaled_action, f, protocol=2)

    # scaled_action = scale_action(best_action, x_range=[-0.44, 0])
    cloth_initial = graph.initial_blanket_state
    initial_covered_status = get_covered_status(all_body_points, cloth_initial)
    fscore = compute_fscore(initial_covered_status, best_covered_status)
    print(scaled_action, best_reward, fscore)

    fig = generate_figure(
            target_limb_code, 
            scaled_action, 
            body_info, 
            all_body_points, 
            cloth_initial = cloth_initial, 
            final_cloths = [best_pred],
            # initial_covered_status = cloth_initial,  # why did i do this initial lol?
            initial_covered_status = initial_covered_status,
            covered_statuses = [best_covered_status] , 
            fscores = [fscore],
            plot_initial=True, compare_subplots=False,
            draw_axes = True)
    # fig.write_image(osp.join(args.pose_dir, f'eval_figure_{fscore:.3f}.png'))
    fig.write_image(osp.join(args.pose_dir, f'eval_figure.png'))
    fig.show()

    eval_data = {
        'action': best_action,
        'scaled_action': scaled_action,
        'cloth_initial': cloth_initial,
        'cloth_final': best_pred,
        'reward': best_reward,
        'fscore': fscore,
        'all_body_points':all_body_points,
        'body_info': body_info,
        'initial_covered_status': initial_covered_status,
        'final_covered_status':best_covered_status,
        'target_limb_code':target_limb_code,
        'is_on_cloth':best_is_on_cloth,
        'best_time':best_time,
        'best_fevals':best_fevals,
        'best_iterations':best_iterations
        }

    with open(osp.join(args.pose_dir, 'cma_eval_data.pkl'), 'wb') as f:
        pickle.dump(eval_data, f)
    
    with open('scaled_action.pkl', 'wb') as f:
        pickle.dump(scaled_action, f, protocol=2)


# %%
