#%%
from cgi import test
import glob, pickle, multiprocessing, math, os, sys
import os.path as osp
import numpy as np
from numpy.lib import index_tricks
import pandas as pd
import math

import torch, torch_geometric
from torch_geometric.data import Dataset, Data

from tqdm import tqdm

# sys.path.insert(1, '/home/kpputhuveetil/git/vBM-GNNs/assistive-gym-fem/assistive_gym/envs')
# from bu_gnn_util import sub_sample_point_clouds


#!!! DELETE CHANGES FOR CMA-DATA
#!! REENABLE USE_CMA_DATA FLAG !!!!!!!!!!!!!
#%%


class BMDataset(Dataset):
    def __init__(self, root, description, transform=None, pre_transform=None, voxel_size=float('NaN'), edge_threshold=0.06, action_to_all=True, 
    use_cma_data=False, testing=False, use_displacement=True, filter_draping = False, use_3D = False, rot_draping=False):
        """
        root is where the data should be stored (data). The directory is split into "raw" and 'preprocessed' directories
        raw folder contains original pickle files
        prprocessed will be filled with the dataset when this class is instantiated
        """
        self.voxel_size = voxel_size
        self.subsample = True if not (np.isnan(self.voxel_size)) else False
        self.edge_threshold = edge_threshold
        self.action_to_all = action_to_all
        self.testing = testing
        self.use_displacement = use_displacement
        self.filter_draping = filter_draping      #! add condition so you can't choose to both rotate and filter
        self.cloth_dim = 2 if not use_3D else 3
        self.rot_draping = rot_draping

        path = os.getcwd()
        data_dir = osp.join(path, root, 'raw/*.pkl')
        #voxel size and edge threshold in cm
        proc_data_dir = f"{description}_vs{self.voxel_size}-et{self.edge_threshold}-aa{int(self.action_to_all)}-disp{int(self.use_displacement)}-c{self.cloth_dim}D"
        root = osp.join(root, proc_data_dir)
        # print(root)

        self.filenames_raw = glob.glob(data_dir)
        # self.filenames_raw = self.filenames_raw[0:3]
        # print(self.filenames_raw)
        self.file_count = len(self.filenames_raw)
        self.num_processes =  multiprocessing.cpu_count()-1
        self.reps = math.ceil(self.file_count/self.num_processes)

        if self.file_count%self.num_processes != 0:
            buffer = [None]*(self.num_processes - (self.file_count%self.num_processes))
            self.filenames = self.filenames_raw + buffer
        else:
            self.filenames = self.filenames_raw

        # # FOR TESTING
        if self.testing:
            self.filenames = self.filenames[0]
            self.num_processes =1
            self.reps = 1
        
        self.unpickling_errors = []
        self.use_cma_data = use_cma_data
        super(BMDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """
        if files exists in raw directory, download is not triggered (download function also not implemented here)
        """
        return self.filenames_raw

    @property
    def processed_file_names(self):
        """
        If these files are found in processed, processing is skipped (don't need to start from scratch)
        Not implemented here
        """
        """ If these files are found in processed_dir, processing is skipped"""
        proc_files = [f'data_{i}.pt' for i in range(len(self.filenames_raw))]
        # print(proc_files[0:10])
        # print(len(proc_files))
        # return [f'data_{i}.pt' for i in range(10)]
        return [f'data_{i}.pt' for i in range(len(self.filenames_raw))]
        # return glob.glob(r'/home/kpputhuveetil/git/bm_gnns/data/processed/*.pt')


    def download(self):
        """
        not implemented
        """
        # # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        # ...
        pass

    def process(self):
        """
        Allows us to construct a graph and pass it to a Data object (which models a single graph) for each data file
        """
        # self.filenames = self.filenames[0:self.num_processes]
        files_array = np.reshape(self.filenames, (self.reps, self.num_processes))
        result_objs = []

        print(self.processed_dir)

        for rep, files in enumerate(tqdm(files_array)):
            # print(f"Rep: {rep+1}, Total Processed: {rep*self.num_processes}")
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                for i,f in enumerate(files):
                    result = pool.apply_async(self.build_graph, args = [f, i+(rep*127)])
                    result_objs.append(result)

                results = [result.get() for result in result_objs]
        
        print("Processing Complete!")


    def build_graph(self, f, idx):
        # global idx
        # data = Data()
        # if not self.testing:
        #     torch.save(data, osp.join(self.processed_dir, f'data_test{idx}.pt'))
        # return


        #! TODO: CHECK THIS!!
        if f is None:
            return

        try:
            raw_data = pickle.load(open(f, "rb"))
        except:
            print('UnpickingError:', f)
            self.unpickling_errors.append(f)
            return 
        
        action = raw_data['action']
        if self.use_cma_data:
            raw_data = raw_data['sim_info']
        human_pose = raw_data['observation'][0]
       

        # initial_num_cloth_points = raw_data['info']['cloth_initial'][0]
        initial_blanket_state = raw_data['info']['cloth_initial'][1]

        # final_num_cloth_points = raw_data['info']['cloth_final'][0]
        final_blanket_state = raw_data['info']['cloth_final'][1]

        if self.rot_draping:
            initial_blanket_state = self.rotate_draping_cloth_points(initial_blanket_state)

        if self.subsample:
            initial_blanket_state, final_blanket_state = self.sub_sample_point_clouds(initial_blanket_state, final_blanket_state)


        #! CHANGE SO THAT CLOTH DIM INFO GETS USED TO DELETE DIM HERE, THEN PASS CORRECT SIZED CLOTH STATE TO FUNCTIONS
        #! DON'T NEED CLOTH DIM ARGS THIS WAY
        # Get  node features
        node_features = self.get_node_features(initial_blanket_state, action, cloth_dim=self.cloth_dim)
        
        edge_indices = self.get_edge_connectivity(initial_blanket_state, cloth_dim=self.cloth_dim)
    
        edge_features = torch.zeros(edge_indices.size()[0], 1, dtype=torch.float)
        cloth_initial, cloth_final = self.get_cloth_as_tensor(initial_blanket_state, final_blanket_state, cloth_dim = self.cloth_dim)

        if self.use_displacement:
            cloth_displacement = self.get_cloth_displacement(cloth_initial, cloth_final)
        else:
            cloth_displacement = False

        # Read data from `raw_path`.
        data = Data(
            x = node_features,
            edge_attr = edge_features,
            edge_index = edge_indices.t().contiguous(),
            cloth_initial = cloth_initial,
            cloth_displacement = cloth_displacement,
            cloth_final = cloth_final,
            action = torch.tensor(action, dtype=torch.float),
            human_pose = torch.tensor(human_pose, dtype=torch.float)
        )

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        if not self.testing:
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    
    def get_node_features(self, cloth_initial, action, cloth_dim):
        """
        returns an array with shape (# nodes, node feature size)
        convert list of lists to tensor
        """
        #! USE SCALE ACTION FUNCtion INSTEAD
        scale = [0.44, 1.05]*2
        action_scaled = action*scale


        if self.action_to_all:
            ## ACTION TO ALL CLOTH POINTS
            nodes = []
            for ind, point in enumerate(cloth_initial):
                node_feature = list(point[0:cloth_dim]) + list(action_scaled)
                nodes.append(node_feature)
                #! USE SOMETHING LIKE THIS INSTEAD
                # nodes = np.append(cloth_initial_3D_pos, [action]*len(cloth_initial_3D_pos), axis=1).tolist()

        else:
            # ACTION ONLY TO GRASPED CLOTH POINTS
            grasp_loc = action_scaled[0:2]
            dist = []
            for i, v in enumerate(cloth_initial):
                v = np.array(v)
                d = np.linalg.norm(v[0:2] - grasp_loc)
                dist.append(d)
            anchor_idx = np.argpartition(np.array(dist), 4)[:4]

            nodes = []
            for ind, point in enumerate(cloth_initial):
                if ind in anchor_idx:
                    node_feature = list(point[0:2]) + list(action_scaled)
                else:
                    node_feature = list(point[0:2]) + [0]*len(action_scaled)
                nodes.append(node_feature)

        return torch.tensor(nodes, dtype=torch.float)
    
    def get_edge_connectivity(self, cloth_initial, cloth_dim):
        """
        returns an array of edge indexes, returned as a list of index tuples
        Data requires indexes to be in COO format so will need to convert via performing transpose (.t()) and calling contiguous (.contiguous())
        """
        cloth_initial = np.array(cloth_initial)
        if cloth_dim == 2:
            cloth_initial = np.delete(cloth_initial, 2, axis = 1)
        threshold = self.edge_threshold
        edge_inds = []
        for p1_ind, point_1 in enumerate(cloth_initial):
            for p2_ind, point_2 in enumerate(cloth_initial): # want duplicate edges to capture both directions of info sharing
                if p1_ind != p2_ind and np.linalg.norm(point_1 - point_2) <= threshold: # don't consider distance between a point and itself, see if distance is within
                    edge_inds.append([p1_ind, p2_ind])
                np.linalg.norm(point_1 - point_2) <= threshold
        # return torch.tensor([0,2], dtype = torch.long)
        return torch.tensor(edge_inds, dtype = torch.long)

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
    
    def get_cloth_displacement(self, cloth_i_tensor, cloth_f_tensor):
        return cloth_f_tensor - cloth_i_tensor


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
        # there's probably a better way to implement this (vectorized), just needed to implement something quick though
        # order of all points in the matrix must be the same before and after rotation
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
        #         if row[0] > 0.44: #! NEED TO THINK ABOUT THIS, THERE ARE POINTS INSIDE THIS BOUNDARY ON THE EDGE TOO
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


        # cloth_right_side = np.array(cloth_right_side).reshape((-1, 3))
        # axis_of_rot = np.broadcast_to([0.44, 0, 0.58], cloth_right_side.shape)
        # cloth_right_side = cloth_right_side - axis_of_rot

        # cloth_right_side_trans = (R_right@cloth_right_side.T).T + axis_of_rot

        # cloth_left_side = np.array(cloth_left_side).reshape((-1, 3))
        # axis_of_rot = np.broadcast_to([-0.44, 0, 0.58], cloth_left_side.shape)
        # cloth_left_side = cloth_left_side - axis_of_rot

        # cloth_left_side_trans = (R_left@cloth_left_side.T).T + axis_of_rot

        # cloth_top = np.array(cloth_top).reshape((-1, 3))

        # # print(cloth_top.shape, cloth_right_side_trans.shape, cloth_left_side_trans.shape)
        # cloth_initial_drap_rot_3D = np.vstack((cloth_top, cloth_right_side_trans, cloth_left_side_trans))

        return cloth_initial_drap_rot_3D


    # ! USE FUNCTION IN BU_GNN_UTIL INSTEAD
    def sub_sample_point_clouds(self, cloth_initial_3D_pos, cloth_final_3D_pos):

        cloth_initial = np.array(cloth_initial_3D_pos)
        cloth_final = np.array(cloth_final_3D_pos)

        if self.filter_draping:
            top_of_bed_points = []
            for i, point in enumerate(cloth_initial):
                if point[2] > 0.58:
                    top_of_bed_points.append(i)
            cloth_initial = cloth_initial[top_of_bed_points]
            cloth_final = cloth_final[top_of_bed_points]


        voxel_size = self.voxel_size
        nb_vox=np.ceil((np.max(cloth_initial, axis=0) - np.min(cloth_initial, axis=0))/voxel_size)
        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((cloth_initial - np.min(cloth_initial, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted=np.argsort(inverse)

        voxel_grid={}
        voxel_grid_cloth_inds={}
        cloth_initial_subsample=[]
        cloth_final_subsample = []
        last_seen=0
        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]= cloth_initial[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            voxel_grid_cloth_inds[tuple(vox)] = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
            
            closest_point_to_barycenter = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
            cloth_initial_subsample.append(voxel_grid[tuple(vox)][closest_point_to_barycenter])
            cloth_final_subsample.append(cloth_final[voxel_grid_cloth_inds[tuple(vox)][closest_point_to_barycenter]])

            last_seen+=nb_pts_per_voxel[idx]

        return cloth_initial_subsample, cloth_final_subsample

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# #%%        
# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_blanket_var_data', 
#     description='blanket_var',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_high_pose_var_data', 
#     description='50k_samples',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

#%%        
# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data', 
#     description='50k_samples',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

#%%
# dataset = BMDataset(
#     root='data_2089', 
#     description='dyn-gnn',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)
#%%
# dataset = BMDataset(root='data_2089', proc_data_dir='edge-thres=2cm_action=GRASP')
# dataset_sub = BMDataset(root='data_2089', proc_data_dir='sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# # print(dataset)
# print(dataset_sub)



# #%%
# ind = 0
# print(dataset[ind])
# print(dataset[ind].edge_index.t().size())
# print(dataset[ind].x.size())
# # print(dataset[ind].y.size())

# len(dataset)
# #%%
# ind = 0
# print(dataset_sub[ind])
# print(dataset_sub[ind].edge_index.t().size())
# print(dataset_sub[ind].x.size())
# # print(dataset[ind].y.size())

# len(dataset_sub)
# #%%
# ratio = []
# for point in dataset:
#     edges = point.edge_index.t().size()[0]
#     nodes = point.x.size()[0]
#     ratio.append(nodes/edges)
# print(np.mean(ratio), np.std(ratio))


# #%%
# print(dataset[ind].x[100])
# # %%
# # %%

# import matplotlib.pyplot as plt
# import numpy as np

# dataset = BMDataset(root='data/')
# # # print(dataset)

# ind = 0

# data_i = dataset[ind].cloth_initial
# x_i = data_i[:,0]
# y_i = data_i[:,1]

# data_f = dataset[ind].cloth_final
# x_f = data_f[:,0]
# y_f = data_f[:,1]

# plt.scatter(x_i, y_i)
# plt.scatter(x_f, y_f)
# plt.show()
# %%