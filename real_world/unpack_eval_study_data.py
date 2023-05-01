#%%
import cv2
import numpy as np
import os.path as osp
import pickle
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
import colorsys
from skimage.color import rgb2hsv
from skimage.io import imread, imshow

usable_subjects = [0, 2, 3, 4, 5, 8, 11]

def isolate_blanket(colors_bgr, show_hsv = False):
    colors_rgb = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)
    colors_rgb_blur = cv2.GaussianBlur(colors_rgb, (11, 11), 0)
    # image = cv2.imread('/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA/subject_TEST/pose_0/uncovered_rgb.png')
    # mask_params = [0.45, 0.535, 0.63]
    mask_params = [0.45, 0.53, 0.63]

    colors_hsv = rgb2hsv(colors_rgb_blur)
    if show_hsv:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(colors_hsv[:,:,0],cmap='hsv')
        ax[0].set_title('hue')
        ax[1].imshow(colors_hsv[:,:,1],cmap='hsv')
        ax[1].set_title('transparency')
        ax[2].imshow(colors_hsv[:,:,2],cmap='hsv')
        ax[2].set_title('value')
        fig.colorbar(imshow(colors_hsv[:,:,0],cmap='hsv')) 
        fig.tight_layout()

    # colors_hsv = np.array([colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]) for rgb in colors_rgb])
    lower_mask = colors_hsv[:,:,0] > mask_params[0]       #* refer to hue channel (value from the colorbar)
    upper_mask = colors_hsv[:,:,0] < mask_params[1]       #* refer to hue channel (value from the colorbar)
    saturation_mask = colors_hsv[:,:,1] > mask_params[2]  #* refer to transparency channel (value from the colorbar)

    mask = upper_mask*lower_mask*saturation_mask

    #* apply segmentation mask to pcd
    red = colors_rgb[:,:,0]*mask
    green = colors_rgb[:,:,1]*mask
    blue = colors_rgb[:,:,2]*mask
    blanket = np.dstack((red,green,blue))
    # plt.figure()
    # imshow(blanket)
    return blanket, colors_rgb

def get_covered_status_rw(all_body_points_px, blanket):
    blanket_2D = blanket[:, :, 1]
    covered_status = []
    for i in range(len(all_body_points_px)):
        is_covered = False
        is_target = all_body_points[i][2]
        y, x = all_body_points_px[i,:]
        px_neighbor = blanket_2D[x-window_size:x+window_size, y-window_size:y+window_size]

        is_covered = True if np.any(px_neighbor > threshold) else False
        
        covered_status.append([is_target, is_covered, (y, x)])
    return covered_status

#%%
# save_dir = '/home/kpputhuveetil/Desktop/paper_figures'
study_data_dir = '/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA'
predef_img_dir = osp.join(study_data_dir, 'EVAL_IMGS/predef')
random_img_dir = osp.join(study_data_dir, 'EVAL_IMGS/random')

subject_ids = [*range(12)]
# subject_ids = [11]
# subject_ind = 10
pose_ind = 0

predef_poses = [[], [], []]
rand_poses = {2:[], 4:[], 5:[], 12:[], 13:[], 14:[], 15:[]}

# for subject_id in [subject_ids[0]]:
for subject_id in subject_ids:
    subject_dir = osp.join(study_data_dir, f'subject_{subject_id}')
    # pose_dir = osp.join(subject_dir,'pose_0_TL2_1674850628')
    # pose_dir = osp.join(subject_dir,'pose_2_TL13_1674855960')
    # pose_dir = osp.join(subject_dir,'pose_6_TL12_1674854123')
    pose_dirs = [x[0] for x in os.walk(subject_dir)][1:]
    print(pose_dirs)
    # pose_dir = pose_dirs[pose_ind]
    # pose_dirs = [pose_dirs[7]]
    fscores = []
    for pose_dir in pose_dirs:
        
        pose_num = int(pose_dir.split('/')[-1].split('_')[1])
        target_limb_code = int(pose_dir.split('/')[-1].split('_')[2][2:])

        if pose_num <= 2:
            save_dir = predef_img_dir
        else:
            save_dir = random_img_dir

        eval_bgr = cv2.imread(osp.join(pose_dir, 'eval_real_rgb.png'))
        cov_bgr = cv2.imread(osp.join(pose_dir, 'covered_rgb_top.png'))

        if eval_bgr is None:
            print(f'IMAGE NOT FOUND: {osp.join(pose_dir, "eval_real_rgb.png")}')
            continue
        if cov_bgr is None:
            print(f'IMAGE NOT FOUND: {osp.join(pose_dir, "covered_rgb_top.png")}')
            continue

        eval_blanket, eval_rgb = isolate_blanket(eval_bgr, show_hsv=False)
        cov_blanket, cov_rgb = isolate_blanket(cov_bgr)

        with open(osp.join(pose_dir,'sim_origin_data.pkl'),'rb') as f:
            data = pickle.load(f)
            dist = data['dist']
            mtx = data['mtx']
            centers_px = data['centers_px']
            centers_m = data['centers_m']
            origin_px = data['origin_px']
            origin_m = data['origin_m']
            m2px_scale = data['m2px_scale']

        with open(osp.join(pose_dir,'human_pose.pkl'), 'rb') as f:
            human_m_correct_axis = pickle.load(f)
        with open(osp.join(subject_dir,'body_info.pkl'),'rb') as f:
            body_info = pickle.load(f)

        all_body_points = get_body_points_from_obs(human_pose=human_m_correct_axis.astype(float), target_limb_code=target_limb_code, body_info=body_info)
        all_body_points[:, [1, 0]] = all_body_points[:, [0, 1]]
        all_body_points_px = (all_body_points[:,:2]*(1/m2px_scale) + origin_px).astype(int)

        if osp.exists(osp.join(pose_dir,'cma_eval_data.pkl')):
            with open(osp.join(pose_dir,'cma_eval_data.pkl'), 'rb') as f:
                scale_action = pickle.load(f)['scaled_action'].reshape((2,2))
                scale_action[:, [1, 0]] = scale_action[:, [0, 1]]
                scale_action_px = (scale_action*(1/m2px_scale) + origin_px).astype(int)
        else:
            continue

        print(scale_action_px)

        window_size = 12
        threshold = 0
        image_cov = cov_rgb.copy()
        image_uncov = eval_rgb.copy()
        covered_status = get_covered_status_rw(all_body_points_px, eval_blanket)
        initial_covered_status = get_covered_status_rw(all_body_points_px, cov_blanket)

        for i in range(len(covered_status)):
            is_target = covered_status[i][0]
            is_covered = covered_status[i][1]
            pos = covered_status[i][2]
            initially_covered = initial_covered_status[i][1]

            plot_point = False
            if save_dir == predef_img_dir:
                plot_point = True

            if is_target == 1:
                infill = (169, 107, 232) if is_covered else (0, 255, 0)
                if infill == (0, 255, 0): plot_point = True
                infill_cov = (169, 107, 232)
            elif is_target == -1: # head points
                # infill = 'red' if is_covered and not is_initially_covered else 'rgba(255,186,71,1)'
                infill = (255, 186, 71) if not is_covered or initially_covered else (255, 0, 0)
                if infill == (255, 0, 0): plot_point = True
                infill_cov = (255, 186, 71)
            else:
                infill = (255, 186, 71) if is_covered or not initially_covered else (255, 0, 0)
                if infill == (255, 0, 0): plot_point = True
                infill_cov = (255, 186, 71)
            
            cv2.circle(image_cov, pos, 6, infill_cov, -1)
            if plot_point:
                cv2.circle(image_uncov, pos, 6, infill, -1)

        fscore = compute_fscore(initial_covered_status, covered_status)

        if pose_num <= 2:
            predef_poses[pose_num].append(fscore)
        else:
            rand_poses[target_limb_code].append(fscore)
            
        if subject_id in usable_subjects:
            if save_dir == predef_img_dir:
                cv2.imwrite(osp.join(save_dir,f'TL{target_limb_code}_subject_{subject_id}_fscore_{round(fscore, 2)}_uncov.png'), cv2.cvtColor(image_uncov, cv2.COLOR_RGB2BGR))
            elif save_dir == random_img_dir:
                # cv2.circle(image_cov, (scale_action_px[1, 0], scale_action_px[1, 1]), 10, (255, 0, 0), -1)
                arrow_tip = 20 / np.linalg.norm(np.array((scale_action_px[0, 0], scale_action_px[0, 1])) - np.array((scale_action_px[1, 0], scale_action_px[1, 1])))
                cv2.circle(image_cov, (scale_action_px[0, 0], scale_action_px[0, 1]), 20, (255, 255, 255), -1)
                cv2.arrowedLine(image_cov, (scale_action_px[0, 0], scale_action_px[0, 1]), 
                                        (scale_action_px[1, 0], scale_action_px[1, 1]), (255, 255, 255), 20, tipLength = arrow_tip)
                cv2.circle(image_cov, (scale_action_px[0, 0], scale_action_px[0, 1]), 15, (0, 0, 0), -1)
                cv2.arrowedLine(image_cov, (scale_action_px[0, 0], scale_action_px[0, 1]), 
                                        (scale_action_px[1, 0], scale_action_px[1, 1]), (0, 0, 0), 10, tipLength = arrow_tip)
                cv2.imwrite(osp.join(save_dir,f'TL{target_limb_code}_subject_{subject_id}_fscore_{round(fscore, 2)}_cov.png'), cv2.cvtColor(image_cov, cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(save_dir,f'TL{target_limb_code}_subject_{subject_id}_fscore_{round(fscore, 2)}_uncov.png'), cv2.cvtColor(image_uncov, cv2.COLOR_RGB2BGR))
        # plt.figure(figsize=(16, 9), dpi=40)
        # plt.title(f'{pose_dir}___{fscore}',fontsize=20)
        # imshow(image_cov)

        # plt.figure(figsize=(16, 9), dpi=40)
        # plt.title(f'{pose_dir}___{fscore}',fontsize=20)
        # imshow(image_uncov)

        # plt.show()

        # fscores.append(fscore)

# print(fscores)
# print(np.mean(fscores))

# %%
predef_stats = [[], []]
for fscores in predef_poses:
    predef_stats[0].append(np.mean(fscores))
    predef_stats[1].append(np.std(fscores))
print(np.array(predef_stats))
# %%
rand_stats = {2:[], 4:[], 5:[], 12:[], 13:[], 14:[], 15:[]}
rand_counts = {2:0, 4:0, 5:0, 12:0, 13:0, 14:0, 15:0}
all_fscores = []
for tl, fscores in rand_poses.items():
    rand_stats[tl].append(np.mean(fscores))
    rand_stats[tl].append(np.std(fscores))
    rand_counts[tl] = len(fscores)
    all_fscores += fscores
print(rand_stats)
print(rand_counts)
print(np.mean(all_fscores), np.std(all_fscores))
# %%
