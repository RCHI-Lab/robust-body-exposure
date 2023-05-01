#%%
from cgi import test
import glob, pickle, multiprocessing, math, os, sys
from logging import root
import os.path as osp
import numpy as np
from numpy.lib import index_tricks
import pandas as pd

import torch, torch_geometric
from torch_geometric.data import Dataset, Data
from pathlib import Path
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import *

#%%

#!! REVERT CHANGES FOR CMA DATA ISSUEs

class Runtime_Graph():
    def __init__(self, root, description, voxel_size=float('NaN'), edge_threshold=0.06, action_to_all=True, cloth_initial=None,
    filter_draping=False, use_3D=False, rot_draping=False):
        self.voxel_size = voxel_size
        self.filter_draping = filter_draping
        self.subsample = True if not (np.isnan(self.voxel_size)) else False
        self.edge_threshold = edge_threshold
        self.action_to_all = action_to_all
        self.cloth_dim = 2 if not use_3D else 3
        self.rot_draping = rot_draping
        # self.testing = testing

        # proc_data_dir = f"{description}_vs{self.voxel_size}-et{self.edge_threshold}-aa{int(self.action_to_all)}"
        # self.proc_data_dir = osp.join(root, proc_data_dir, 'processed')
        # Path(self.proc_data_dir).mkdir(parents=True, exist_ok=True)
        # if self.rot_draping:
        #     self.initial_blanket_state = self.rotate_draping_cloth_points(cloth_initial)
        # if self.subsample:
        #     self.initial_blanket_state = self.sub_sample_point_clouds(cloth_initial)

        if self.rot_draping:
            cloth_initial = self.rotate_draping_cloth_points(cloth_initial)
        if self.subsample:
            cloth_initial = self.sub_sample_point_clouds(cloth_initial)
        
        self.initial_blanket_state = cloth_initial

            
        self.edge_indices = get_edge_connectivity(self.initial_blanket_state, self.edge_threshold, self.cloth_dim)
        if self.cloth_dim == 2:
            self.initial_blanket_state = np.delete(np.array(self.initial_blanket_state), 2, axis = 1)
        
        self.edge_features = torch.zeros(self.edge_indices.size()[0], 1, dtype=torch.float)
        self.global_vector = torch.zeros(1, 0, dtype=torch.float32)


    def build_graph(self, action):

        node_features = self.get_node_features(self.initial_blanket_state, action)

        data = Data(
            x = node_features,
            edge_attr = self.edge_features,
            edge_index = self.edge_indices.t().contiguous(),
            batch = torch.zeros(node_features.shape[0], dtype=torch.long),
            cloth_initial = None,
            cloth_final = None,
            action = None,
            human_pose = None
        )

        return data
    
    #!! REPLACE WITH BU GNN FUNCTIONS
    
    def get_node_features(self, cloth_initial, action):
        """
        returns an array with shape (# nodes, node feature size)
        convert list of lists to tensor
        """
        #! REPLACE WITH FUNCTION
        scale = [0.44, 1.05]*2
        action_scaled = action*scale


        if self.action_to_all:
            nodes = np.append(cloth_initial, [action]*len(cloth_initial), axis=1).tolist()

        return torch.tensor(nodes, dtype=torch.float)

    def get_rotation_matrix(self, axis, theta):
            """
            Find the rotation matrix associated with counterclockwise rotation
            about the given axis by theta radians.
            Credit: http://stackoverflow.com/users/190597/unutbu

            Args:
                axis (list): rotation axis of the form [x, y, z]
                theta (float): rotational angle in radians

            Returns:
                array. Rotation matrix.
            """

            axis = np.asarray(axis)
            theta = np.asarray(theta)
            axis = axis/math.sqrt(np.dot(axis, axis))
            a = math.cos(theta/2.0)
            b, c, d = -axis*math.sin(theta/2.0)
            aa, bb, cc, dd = a*a, b*b, c*c, d*d
            bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

            return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    def rotate_draping_cloth_points(self, cloth_initial_3D_pos):
        axis = [0, 1, 0]
        theta = np.pi/2
        R_right = self.get_rotation_matrix(axis, -theta)
        R_left = self.get_rotation_matrix(axis, theta)

        axis_rot_r = [0.44, 0, 0.58]
        axis_rot_l = [-0.44, 0, 0.58]

        new_cloth = []
        for row in np.array(cloth_initial_3D_pos):
            if row[2] < 0.575:
                if row[0] > 0: # keep an eye on this - is this working better than 0.44?
                    row = R_right @ (row - axis_rot_r) + axis_rot_r
                elif row[0] < 0:
                    row = R_left @ (row - axis_rot_l) + axis_rot_l
            new_cloth.append(row)

        cloth_initial_drap_rot_3D = np.array(new_cloth)
        # cloth_right_side = []
        # cloth_left_side = []
        # cloth_top = []
        # for row in cloth_initial_3D_pos:
        #     if row[2] < 0.58:
        #         if row[0] > 0.44:
        #             cloth_right_side.append(row)
        #         elif row[0] < -0.44:
        #             cloth_left_side.append(row)
        #         else:
        #             cloth_top.append(row)
        #     else:
        #         cloth_top.append(row)

        # axis = [0, 1, 0]
        # theta = np.pi/2
        # R_right = self.get_rotation_matrix(axis, -theta)
        # R_left = self.get_rotation_matrix(axis, theta)


        # cloth_right_side = np.array(cloth_right_side).reshape((-1,3))
        # axis_of_rot = np.broadcast_to([0.44, 0, 0.58], cloth_right_side.shape)
        # cloth_right_side = cloth_right_side - axis_of_rot

        # cloth_right_side_trans = (R_right@cloth_right_side.T).T + axis_of_rot

        # cloth_left_side = np.array(cloth_left_side).reshape((-1,3))
        # axis_of_rot = np.broadcast_to([-0.44, 0, 0.58], cloth_left_side.shape)
        # cloth_left_side = cloth_left_side - axis_of_rot

        # cloth_left_side_trans = (R_left@cloth_left_side.T).T + axis_of_rot

        # cloth_top = np.array(cloth_top).reshape((-1,3))

        # # print(cloth_top.shape, cloth_right_side_trans.shape, cloth_left_side_trans.shape)
        # cloth_initial_drap_rot_3D = np.vstack((cloth_top, cloth_right_side_trans, cloth_left_side_trans))

        return cloth_initial_drap_rot_3D

    

    # ! USE FUNCTION IN BU_GNN_UTIL INSTEAD
    def sub_sample_point_clouds(self, cloth_initial_3D_pos):
        cloth_initial = np.array(cloth_initial_3D_pos)

        if self.filter_draping:
            top_of_bed_points = []
            for i, point in enumerate(cloth_initial):
                if point[2] > 0.58:
                    top_of_bed_points.append(i)
            cloth_initial = cloth_initial[top_of_bed_points]

        voxel_size = self.voxel_size
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

    # ! USE EXISTING FUNCTION INSTEAAD
    def get_cloth_as_tensor(self, cloth_initial_3D_pos, cloth_final_3D_pos, cloth_dim):
        if cloth_dim == 2:
            cloth_initial_pos = np.delete(np.array(cloth_initial_3D_pos), 2, axis = 1)
            cloth_final_pos = np.delete(np.array(cloth_final_3D_pos), 2, axis = 1)
        elif cloth_dim == 3:
            cloth_initial_pos = np.array(cloth_initial_3D_pos)
            cloth_final_pos = np.array(cloth_final_3D_pos)

        cloth_i_tensor = torch.tensor(cloth_initial_pos, dtype=torch.float)
        cloth_f_tensor = torch.tensor(cloth_final_pos, dtype=torch.float)
        return cloth_i_tensor, cloth_f_tensor
    
    # def dict_to_Data(self, data_dict):
    #     data = Data(
    #         x = data_dict['x'],
    #         edge_attr =  data_dict['edge_attr'],
    #         edge_index =  data_dict['edge_index'],
    #         cloth_initial = data_dict['cloth_initial'],
    #         cloth_final =  data_dict['cloth_final'],
    #         action =  data_dict['action'],
    #         human_pose =  data_dict['human_pose']
    #     )
    #     return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
