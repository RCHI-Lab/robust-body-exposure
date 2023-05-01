#%%
import os

from bm_dataset import BMDataset
from gnn_manager import GNN_Manager
import torch

dataset_dir = '/home/kpputhuveetil/git/vBM-GNNdev/DATASETS/standard'

datasets = [0, 0, 0, 0, 0, 0, 0, 0]
standard_data = BMDataset(
        root=dataset_dir, 
        description='rotate_overhang',
        voxel_size=0.05, edge_threshold=0.06,
        rot_draping=True)
standard_data_3D = BMDataset(
        root=dataset_dir, 
        description='3D_representation',
        voxel_size=0.05, edge_threshold=0.06,
        rot_draping=False, use_3D=True)
#%%
# names = ['filt_drape', '2D', 'rot_drape', '3D']
# names = ['standard_2D_10k', 'standard_2D_50k', 'pose_var_2D_50k', 'blanket_var_2D_50k', 'blanket_var_3D_50k', 'body_var_2D_50k', 'no_human_50k', 'combo_var']
# dataset_sizes = [10000, 50000, 50000, 50000, 50000, 50000, 50000, 50000]

model_names = [
        'standard_3D_10k'
        'body+blanket_var_5k'
        'combo_var_20k', 
        'standard_2D_500', 
        'standard_2D_100',
        'standard_2D_25k',
        'pose_var_2D_10k', 
        'blanket_var_2D_10k', 
        'blanket_var_3D_10k', 
        'body_var_2D_10k', 
        'no_human_10k',
        'throwaway']
dataset_sizes = [10000, 500, 100, 25000, 10000, 10000, 10000, 10000, 10000, 1000]
datasets = [standard_data_3D]


for i in range(len(datasets)):
        initial_dataset = datasets[i]
        initial_dataset = initial_dataset[:dataset_sizes[i]+100]

        torch.cuda.empty_cache()
        # device = 'cuda:0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_manager = GNN_Manager(device)
        gnn_manager.set_initial_dataset(initial_dataset, (dataset_sizes[i], 100))
        save_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'trained_models/FINAL_MODELS'))
        model_description = f'{model_names[i]}_{dataset_sizes[i]}'
        train_test = (10000, 100) #! not used atm
        num_images = 100
        epochs = 250
        proc_layers = 4
        learning_rate = 1e-4
        seed = 1001
        batch_size = 100
        num_workers = 4
        use_displacement = True


        gnn_manager.initialize_new_model(save_dir, train_test, 
                proc_layers, num_images, epochs, learning_rate, seed, batch_size, num_workers,
                model_description, use_displacement)
        gnn_manager.set_dataloaders()
        gnn_manager.train(epochs)