#%%
import math
import sys
import time
import argparse

# import multiprocessing
from torch import multiprocessing

sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
import os.path as osp
from pathlib import Path

import cma
import numpy as np
import torch
from assistive_gym.envs.bu_gnn_util import *
from assistive_gym.learn import make_env
from build_runtime_graph import Runtime_Graph
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *
from gnn_manager import GNN_Manager
from gym.utils import seeding
from tqdm import tqdm


#%%

#* parameters for the graph representation of the cloth fed as input to the model
all_graph_configs = {
        '2D':{'filt_drape':False,  'rot_drape':True, 'use_3D':False, 'use_disp':True},
        '3D':{'filt_drape':False,  'rot_drape':False, 'use_3D':True, 'use_disp':True}
    }
#* parameters for which enviornmental variations to use in simulation
all_env_vars = {
        'standard':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':False},   # standard
        'body_shape_var':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':True},    # body shape var
        'pose_var':{'blanket_var':False, 'high_pose_var':True, 'body_shape_var':False},    # high pose var
        'blanket_var':{'blanket_var':True, 'high_pose_var':False, 'body_shape_var':False},    # blanket var
        'combo_var':{'blanket_var':True, 'high_pose_var':True, 'body_shape_var':True}       # combo var
    }


# check if grasp is on the cloth BEFORE subsampling! cloth_initial_raw is pre subsampling
# ! maybe there is already a util function for this? check_grasp_on_cloth
def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D):
    action = scale_action(action)
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

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[1]:.2f}, Sim Reward: {output[2]:.2f}, CMA Time: {output[3]/60:.2f}, TL: {output[4]}, GoC: {output[5]}")
    
    # print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

#%%
def gnn_cma(env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_var, max_fevals):

    use_disp = graph_config['use_disp']
    filter_draping = graph_config['filt_drape']
    rot_draping = graph_config['rot_drape']
    use_3D = graph_config['use_3D']

    coop = 'Human' in env_name
    seed = seeding.create_seed()
    env = make_env(env_name, coop=coop, seed=seed)

    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = env_var['blanket_var'],
        high_pose_var = env_var['high_pose_var'],
        body_shape_var = env_var['body_shape_var'])
    done = False
    # #env.render())
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))
    if target_limb_code is None:
        target_limb_code = randomize_target_limbs()
    cloth_initial_raw = env.get_cloth_state()
    env.set_target_limb_code(target_limb_code)
    pop_size = 8
    # return seed, env.target_limb_code, human_pose[1, 0], idx

    # f = eval_files[0]
    # raw_data = pickle.load(open(f, "rb"))
    # cloth_initial = raw_data['info']['cloth_initial'][1]
    # human_pose = raw_data['observation'][0]
    # human_pose = np.reshape(raw_data['observation'][0], (-1,2))
    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    graph = Runtime_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        cloth_initial=cloth_initial_raw,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D)

    # print('graph constructed')

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': pop_size, 'maxfevals': max_fevals, 'tolfun': 1e-11, 'tolflatfitness': 20, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'max_fevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)

    
    x0 = set_x0_for_cmaes(target_limb_code)
    
    # x0 = [0.5, 0.5, -0.5, -0.5]     # * grasp scaled is at [0.22, 0.525]
    # x0 = [0, 0, 0, -0.5]
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.2
    reward_threshold = 95

    total_fevals = 0

    fevals = 0
    iterations = 0
    t0 = time.time()

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
        output = [cost_function(x, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        is_on_cloth = output[3]
        # print(-1*np.array(costs))
        es.tell(actions, costs)
        
        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_status[best_cost_ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
            best_is_on_cloth = is_on_cloth[best_cost_ind]
        if best_reward >= reward_threshold:
            break
    observation, env_reward, done, info = env.step(best_action)     
    # print(info.keys())

    # return cloth_initial, best_pred, all_body_points, best_covered_status, info
    sim_info = {'observation':observation, 'reward':env_reward, 'done':done, 'info':info}
    cma_info = {
        'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'best_time':best_time,
        'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations}
    
    # only save data is error is greater than some threshold, 15
    save_data_to_pickle(
        idx, 
        seed, 
        best_action, 
        human_pose, 
        target_limb_code,
        sim_info,
        cma_info,
        iter_data_dir)
    # save_dataset(idx, graph, best_data, sim_info, best_action, human_pose, best_covered_status)
    return seed, best_reward, env_reward, best_time, target_limb_code, best_is_on_cloth


def evaluate_dyn_model(env_name, target_limb_code, trials, model, iter_data_dir, device, num_processes, graph_config, env_variations, max_fevals):

    result_objs = []
    # ! Why doing trials/num_processes? equals 1
    for j in tqdm(range(math.ceil(trials/num_processes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                idx = i+(j*num_processes)
                result = pool.apply_async(gnn_cma, args = (env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_variations, max_fevals), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            all_results.append(results)
    
    results_array = np.array(results)
    pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])

    return list(results_array[:,1]), list(results_array[:,2]), list(pred_sim_reward_error)


#%%

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    trained_models_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/GNN'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval-multiple-models', type=bool, default=False)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--graph-config', type=str)
    parser.add_argument('--env-var', type=str)
    parser.add_argument('--max-fevals', type=int, default=300)
    parser.add_argument('--num-rollouts', type=int, default=500)
    args = parser.parse_args()

    if not args.eval_multiple_models:
        
        loop_data = [{ 
            'model': args.model_path,
            'graph_config':args.graph_config,
            'env_var':args.env_var,
            'max_fevals':args.max_fevals
        }]
    
    else:

        loop_data = [
            {'model': 'standard_2D_10k_epochs=250_batch=100_workers=4_1668718872', 'graph_config': '2D', 'env_var': 'standard', 'max_fevals':300},
            {'model': 'standard_3D_10k_epochs=250_batch=100_workers=4_1675341883', 'graph_config': '3D', 'env_var': 'combo_var', 'max_fevals':300},
            {'model': 'standard_3D_10k_epochs=250_batch=100_workers=4_1675341883', 'graph_config': '3D', 'env_var': 'standard', 'max_fevals':150},
            {'model': 'standard_3D_10k_epochs=250_batch=100_workers=4_1675341883', 'graph_config': '3D', 'env_var': 'combo_var', 'max_fevals':150},

        ]

    env_name = "RobustBodyExposure-v1"
    target_limb_code = None

    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint= osp.join(trained_models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        graph_config = all_graph_configs[data['graph_config']]
        print(graph_config)
        max_fevals = data['max_fevals']

    
        data_dir = osp.join(checkpoint, f'cma_evaluations/{env_var}_{max_fevals}_{round(time.time())}')
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(data_dir)

        device = 'cpu'
        gnn_manager = GNN_Manager(device)
        gnn_manager.load_model_from_checkpoint(checkpoint)
        gnn_manager.model.to(torch.device('cpu'))
        gnn_manager.model.share_memory()
        gnn_manager.model.eval()

        counter = 0
        all_results = []

        # reserve one cpu to keep working while collecting data
        num_processes = multiprocessing.cpu_count() - 1



        iterations = 10

        num_processes = trials = 50

        iterations = round(args.num_rollouts/num_processes)

        for iter in tqdm(range(iterations)):

            cma_reward, sim_reward, pred_sim_reward_error = evaluate_dyn_model(
                env_name=env_name,
                target_limb_code = target_limb_code,
                trials = trials,
                model = gnn_manager.model, 
                iter_data_dir = data_dir, 
                device = device,
                num_processes = num_processes, 
                graph_config = graph_config,
                env_variations = env_variations,
                max_fevals=max_fevals)

    print("ALL EVALS COMPLETE")