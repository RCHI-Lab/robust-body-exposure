#%%
import glob
import os.path as osp
import pickle
import sys
from pathlib import Path

import numpy as np
from cma_gnn_util import *

sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
import matplotlib.pyplot as plt
from assistive_gym.envs.bu_gnn_util import *

# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_1k_epochs=250_batch=100_workers=4_1670332079'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_5k_epochs=250_batch=100_workers=4_1670298942'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_7.5k_epochs=250_batch=100_workers=4_1670300908'
#model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_10k_epochs=250_batch=100_workers=4_1668718872'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_25k_epochs=250_batch=100_workers=4_1670303804'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_50k_epochs=250_batch=100_workers=4_1668722725'
model_path = '/home/kendragivens/env/robust-body-exposure'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/no_human_10k_epochs=250_batch=100_workers=4_1670329271'
# model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/combo_var_10k_epochs=250_batch=100_workers=4_1670294467'
#eval_dir_name = 'cma_evaluations'
# eval_dir_name = 'random_search_evals'
# eval_dir_name = 'parallel_temp_evaluations'
#eval_condition = 'combo_var_300'
eval_dir_name = 'assistive-gym-fem/assistive_gym'

eval_condition = 'New_Anchors_Fixed_Actions'
# eval_condition = 'New_Anchors_Rand_Actions'
# eval_condition = 'Same_Anchors_Fixed_Actions'
# eval_condition = 'New_Anchors_Fixed_Actions'

# data_path = osp.join(model_path, eval_dir_name, eval_condition, 'raw/*.pkl')
# image_dir = osp.join(model_path, eval_dir_name, eval_condition, 'images')

data_path = osp.join(model_path, eval_dir_name, eval_condition, 'raw/*.pkl')
image_dir = osp.join(model_path, eval_dir_name, eval_condition, 'images')

Path(image_dir).mkdir(parents=True, exist_ok=True)

filenames = glob.glob(data_path)

#filenames_sim_dyn = glob.glob('/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_10k_epochs=250_batch=100_workers=4_1668718872/cma_evaluations/standard_150/sim_dyn_eval/*.pkl')

#sim_dyn_seeds = [fsd.split('/')[-1].split('_')[2] for fsd in filenames_sim_dyn]
# print(filenames)
count = 0
# # filename = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/one_step_improv_new_epochs=500_batch=100_workers=4_1658469685/cma_evaluation_high_pose_var/raw/tl1_c6_15576168541647790998_pid51932.pkl'
# # filename = filenames[12]

# f = open(filename, 'rb')
# raw_data = pickle.load(f)
# target_limb_code = raw_data['target_limb_code']
# human_pose = raw_data['human_pose']

# cloth_initial = raw_data['sim_info']['info']['cloth_initial_subsample']
# pred = raw_data['cma_info']['best_pred']
# all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code)
# initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis = 1))
# covered_status = raw_data['cma_info']['best_covered_status']
# info = raw_data['sim_info']['info']
# cma_reward = raw_data['cma_info']['best_reward']
# sim_reward = raw_data['sim_info']['reward']

# # #!!! CHANGE TO REDUCE MARGIN SIZE OR BIGGER TEXT
# fig = generate_figure_sim_results(cloth_iniStial, pred, all_body_points, covered_status, initial_covered_status, info, cma_reward, sim_reward)

# FIX TO PRESENT BODY SHAPE VARIATION!

##%%
# filenames = filenames[7:10]
for i, filename in enumerate(filenames):
    with open(filename, 'rb') as f:

        seed = filename.split('/')[-1].split('_')[2]
        # if seed not in sim_dyn_seeds: 
        #     continue

        raw_data = pickle.load(f)
        # print(raw_data)

        if 'cloth_initial' not in raw_data['info']:
            continue

        tl = filename.split('/')[-1]
        fig_id = tl.split('_')[2]
        tl = tl.split('_')[0]

        tl_dir = osp.join(image_dir, tl)
        Path(tl_dir).mkdir(parents=True, exist_ok=True)

        cloth_initial = np.array(raw_data['info']['cloth_initial'][1])
        cloth_intermediate = np.array(raw_data['info']['cloth_intermediate'][1])
        cloth_final = np.array(raw_data['info']['cloth_final'][1])
        action = scale_action(raw_data['action'])
        all_body_points = raw_data['info']['all_body_points']

        fig = generate_figure(
            action, 
            all_body_points, 
            cloth_initial, 
            cloth_intermediate,
            cloth_final,
            plot_initial=True, compare_subplots=True)
        
        img_file = f'{eval_condition}_{seed}.png'
        # fig.show()
        fig.write_image(osp.join(image_dir, img_file))

        # fig.write_image(osp.join(tl_dir, img_file))
        # fig.write_image(osp.join(model_path, eval_dir_name, eval_condition, 'eval_imgs', img_file))
 
            

 # %%
