#%%
# from _typeshed import NoneType
import configparser
import glob
import os.path as osp
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from bm_dataset import BMDataset
# from bm_dataset_3D import BMDataset
# from bm_dataset_no_edge_attr import BMDataset
from models_graph_res import GNNModel
from requests import delete
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm import tqdm


#%%
class GNN_Manager():
    def __init__(self, device):
        self.set_device(device)
    
    def set_device(self, device):
        self.device = torch.device(device)
        print("Device:", self.device)
    
    def set_args(self, model_config):
        proc_layers = int(model_config['proc_layer_num'])
        learning_rate = float(model_config['learning_rate'])
        seed = int(model_config['seed'])
        global_size = int(model_config['global_size'])
        output_size = int(model_config['output_size'])
        node_dim = int(model_config['node_dim'])
        edge_dim = int(model_config['edge_dim'])

        self.args = SimpleNamespace(seed=seed,
                                    learning_rate=learning_rate,
                                    proc_layer_num=proc_layers, 
                                    global_size=global_size,
                                    output_size=output_size,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim)

    def set_training_config(self, initial_dataset=None, train_test=None, proc_layers=None, global_size=0, epochs=None, learning_rate=None, seed=None):
        node_dim = initial_dataset[0].x.shape[1]
        edge_dim = initial_dataset[0].edge_attr.shape[1]
        output_size = initial_dataset[0].cloth_final.shape[1]
        global_size = global_size
        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['Dataset'] = {
            'dir':initial_dataset.root,
            'train_test':train_test,
            'num_total_data':len(initial_dataset),
            'num_train_data':len(self.TRAIN_DATASET),
            'num_test_data':len(self.TEST_DATASET),
            'voxel_size':initial_dataset.voxel_size,
            'subsample':initial_dataset.subsample,
            'edge_threshold':initial_dataset.edge_threshold,
            'action_to_all':initial_dataset.action_to_all,
            'shuffle':False}
        config['Model'] = {
            'seed':seed,
            'learning_rate':learning_rate,
            'epochs':epochs,
            'proc_layer_num':proc_layers, 
            'global_size':global_size,
            'output_size':output_size,
            'node_dim':node_dim,
            'edge_dim':edge_dim}
        with open(self.config_dir, 'w') as configfile:
            config.write(configfile)
    
    #! COME BACK AND FINALIZE THIS
    def update_training_config(self, config, iter):
        config['Continual Learning Stats'][f'iteration {iter}'] = {
            'new_dataset_dir': [],
            'num_train_data_added':[],
        }
        with open(self.config_dir, 'w') as configfile:
            config.write(configfile)
    

    def set_initial_dataset(self, dataset, train_test=None):
        # if you pass a single float for train_test, interpret as the ratio of train:test points to use from the entire dataset
        if isinstance(train_test, float):
            train_test_ratio = train_test
            train_len = round(len(dataset)*train_test_ratio)
            test_len = -1
        # if you pass a tuple for tr--9in_test, interpret as the number of train points and test points to select from the entire dataset of points
        elif isinstance(train_test, tuple): 
            train_len = train_test[0]
            test_len = train_test[0] + train_test[1]
        self.TRAIN_DATASET = self.initial_dataset = dataset[:train_len]
        self.TEST_DATASET = dataset[train_len:test_len]
        self.dataset_dir = [dataset.root]

        # TRAIN_DATASET = TEST_DATASET = [dataset[0]]
        # batch_size = num_workers = 1


    def set_dataloaders(self):
        self.trainDataLoader = torch_geometric.loader.DataLoader(self.TRAIN_DATASET, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                                        pin_memory=True, drop_last=False)
        self.testDataLoader = torch_geometric.loader.DataLoader(self.TEST_DATASET, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                                        pin_memory=True, drop_last=False)
        # self.imageDataLoader = torch_geometric.loader.DataLoader(self.TEST_DATASET[0:num_images], batch_size=1, shuffle=False, num_workers=1,
        #                                                 pin_memory=True, drop_last=False)

        print("The number of training data is: %d" % len(self.TRAIN_DATASET))
        print("The number of test data is: %d" % len(self.TEST_DATASET))
        # print("The number of image data is: %d" % num_images)
    

    def add_to_train_set(self, new_dataset):

        self.TRAIN_DATASET = torch.utils.data.ConcatDataset([self.TRAIN_DATASET, new_dataset])
        self.dataset_dir.append(new_dataset.root)
        # self.trainDataLoader = torch_geometric.loader.DataLoader(self.TRAIN_DATASET, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
        #                                                     pin_memory=True, drop_last=False)
        # # self.batch_size = len(dataset)
        # self.batch_size = 1

    def load_model_from_checkpoint(self, model_dir, model_checkpoint_number=None, batch_size=None, num_workers=None, new_save_dir=None):
        config = configparser.ConfigParser()

        print('Model Config:', osp.join(model_dir, 'config.ini'))
        config.read(osp.join(model_dir, 'config.ini'))
        model_config = config['Model']
        self.set_args(model_config)
        #! Get rid of the model_config variable?????? just use config['Model']

        self.batch_size = batch_size if batch_size != None else int(model_config['batch_size'])
        self.num_workers = num_workers if num_workers != None else int(model_config['num_workers'])
        #! CHANGE THIS BACK IS POSSIBLE
        self.num_workers = 0

        self.load_model(model_dir, model_checkpoint_number=model_checkpoint_number)

        if new_save_dir is not None:
            self.set_new_save_dir(new_save_dir)
        else:
            self.set_new_save_dir(model_dir)
    
    def load_model(self, model_dir, model_checkpoint_number=None):
        checkpoint_dir = osp.join(model_dir, 'checkpoints')
        
        # data_dir = osp.join(path, root, 'raw/*.pkl')
        if model_checkpoint_number==None:
            all_checkpoints = glob.glob(osp.join(checkpoint_dir, '*.pth'))
            checkpoint_path = sorted(all_checkpoints)[-1] # pick last checkpoint
            model_checkpoint_number = len(all_checkpoints)-1
            # print(model_checkpoint_number)
        else:
            checkpoint_path = osp.join(checkpoint_dir, f'model_{model_checkpoint_number}.pth')

        self.model_checkpoint_number = model_checkpoint_number + 1 #! add code to generate this when you first make the model too


        self.model = GNNModel(self.args, self.args.proc_layer_num, self.args.global_size, self.args.output_size)
        self.model.to(self.device)
        if self.device == torch.device('cpu'):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model'])
        else:
            self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model_criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=3, verbose=True)
    
    def delete_model(self):
        del self.model
    
    def set_new_save_dir(self, new_save_dir):
        self.writer_dir = osp.join(new_save_dir, 'runs')
        self.writer = SummaryWriter(self.writer_dir)

        self.checkpoints_dir = osp.join(new_save_dir, 'checkpoints')
        Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    def initialize_new_model(self, save_dir, train_test=None, proc_layers=None, num_images=None, epochs=None, learning_rate=None, seed=None, batch_size=None, num_workers=None, model_descirption=None, use_displacement=False):
        torch.cuda.empty_cache()
        # print(f"TEST: edge threshold = {edge_thres}, action to {action_to_node} nodes, processing layers = {proc_layers}")
        self.model_id = f"{model_descirption}_epochs={epochs}_batch={batch_size}_workers={num_workers}_{round(time.time())}"
        
        self.set_new_save_dir(osp.join(save_dir, self.model_id))
        print(self.checkpoints_dir)

        self.config_dir = osp.join(save_dir, self.model_id, 'config.ini')
        print('Model Config:', self.config_dir)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # # dataset = dataset.shuffle()

        # self.set_initial_dataset(dataset)
        # self.set_dataloaders(batch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model_checkpoint_number = 0
        self.use_displacement = use_displacement

        dataset = self.initial_dataset
        node_dim = dataset[0].x.shape[1]
        edge_dim = dataset[0].edge_attr.shape[1]
        output_size = dataset[0].cloth_final.shape[1]
        global_size = 0 #! Don't Hardcode

        print(f"Node feature length: {node_dim}, edge feature length: {edge_dim}, global_size: {global_size}, output size: {output_size}")
        print("Processing layers:", proc_layers)

        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['Dataset'] = {
            'dir':self.dataset_dir,
            # 'train_test':train_test,
            # 'num_total_data':len(dataset),
            'num_train_data':len(self.TRAIN_DATASET),
            'num_test_data':len(self.TEST_DATASET),
            'voxel_size':dataset.voxel_size,
            'subsample':dataset.subsample,
            'edge_threshold':dataset.edge_threshold,
            'action_to_all':dataset.action_to_all,
            'shuffle':False}
        config['Model'] = {
            'seed':seed,
            'learning_rate':learning_rate,
            'epochs':epochs,
            'proc_layer_num':proc_layers, 
            'global_size':global_size,
            'output_size':output_size,
            'node_dim':node_dim,
            'edge_dim':edge_dim,
            'batch_size':batch_size,
            'num_workers':num_workers,
            'use_displacement':self.use_displacement}
        with open(self.config_dir, 'w') as configfile:
            config.write(configfile)

        self.set_args(config['Model'])

        self.model = GNNModel(self.args, self.args.proc_layer_num, self.args.global_size, self.args.output_size)
        self.model.to(self.device)
        self.model_criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=3, verbose=True)


        print("Set up complete")
    
    def run(self, args, epoch, dataloader, mode, take_images=False, fig_dir=None):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()

        total_state_loss = 0
        state_relative_error_gt = 0
        state_rmse_error_gt = 0


        with torch.set_grad_enabled(mode == 'train'):
            for i, data in enumerate(dataloader):
                if mode == 'train':
                    self.optimizer.zero_grad()

                data = data.to(self.device).to_dict()
                batch = data['batch']
                batch_num = np.max(batch.data.detach().cpu().numpy()) + 1
                global_vec = torch.zeros(batch_num, args.global_size, dtype=torch.float32, device=self.device)
                data['u'] = global_vec
                #! ADD THIS WHEN RUNNING NO EDGE ATTR DATASETS
                # edge_attr = torch.zeros(data['edge_index'].size()[1], 1, dtype=torch.float32, device=self.device)
                # data['edge_attr'] = edge_attr
                # data['edge_attr'] = data['edge_attr'].float()

                if self.use_displacement:
                    state_target = data['cloth_displacement']
                else:
                    state_target = data['cloth_final']

                state_predicted = self.model(data)['target']
                # out = (state_target, state_predicted)

                # L1 norm for MSE
                # state_predicted = state_predicted.contiguous().view(-1, 1)
                # state_target = state_target.contiguous().view(-1, 1)
                
                # L2 norm for MSE
                state_predicted = state_predicted.contiguous()
                state_target = state_target.contiguous()

                state_loss = self.model_criterion(state_predicted, state_target)
                
                
                total_state_loss += state_loss.detach().item()

                state_pred_numpy = state_predicted.data.detach().cpu().numpy().flatten()
                state_target_numpy = state_target.data.detach().cpu().numpy().flatten()

                state_relative_error_gt += np.mean(np.abs(state_pred_numpy - state_target_numpy) / (np.abs(state_target_numpy) + 1e-10))
                state_rmse_error_gt += np.sqrt(np.mean((state_pred_numpy - state_target_numpy) ** 2))

                if mode == 'train':
                    state_loss.backward()
                    self.optimizer.step()
                
                total_state_loss += state_loss.item()

                if take_images:
                    figure = self.old_generate_eval_figure(data, state_predicted)
                    figure.savefig(osp.join(fig_dir, f'eval_{i}.png'))
                    # plt.show()
                    plt.close()
            
            if mode == 'train':
                self.scheduler.step(total_state_loss / len(dataloader))
            


        eval_metrics = {
            'total_loss':total_state_loss,
            'rmse': state_rmse_error_gt,
            'relative_error': state_relative_error_gt}

        return eval_metrics
        # return (eval_metrics, out)

    def train(self, epochs):
        self.model.to(self.device)
        best_loss = best_rmse = best_rele = best_time = None

        t_initial = time.time()
        for epoch in tqdm(range(epochs)):
            elapsed_epoch = self.model_checkpoint_number+epoch
            t0 = time.time()
            take_images = False
            train_metrics = self.run(
                                    self.args, 
                                    epoch, 
                                    self.trainDataLoader, 
                                    'train',
                                    take_images=False)
            t1 = time.time()
            self.writer.add_scalar("Loss/train", train_metrics['total_loss'], elapsed_epoch)
            self.writer.add_scalar("RMSE/train", train_metrics['rmse'], elapsed_epoch)
            self.writer.add_scalar("Relative_error/train", train_metrics['relative_error'], elapsed_epoch)
            self.writer.add_scalar("Time_per_epoch/train", t1-t0, elapsed_epoch)

            if (best_loss is None) or (train_metrics['total_loss'] < best_loss):
                best_loss = train_metrics['total_loss']
                best_rmse = train_metrics['rmse']
                best_rele = train_metrics['relative_error']
                best_time = t1-t0

            # print(train_metrics)
            torch.cuda.empty_cache()

            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': elapsed_epoch,
                'loss': train_metrics['total_loss']
            }
            savepath = osp.join(self.checkpoints_dir, 'model_{}.pth'.format(elapsed_epoch))
            torch.save(save_dict, savepath)

        self.writer.add_hparams(
        {'proc_layers': self.args.proc_layer_num},
        {  
            'best_loss':best_loss,
            'best_rmse':best_rmse,
            'best_rele':best_rele,
            'best_time':best_time
        },
        run_name='hparams'
        )
        self.writer.flush()
        self.writer.close()

        print('Training Done!\n')
    
    def evaluate(self, checkpoint_path):

        run_dir = "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/runs"
        fig_dir = osp.join(run_dir, 'images')
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        eval_metrics = self.run(
                                self.args, 
                                None, 
                                self.testDataLoader, 
                                'eval',
                                take_images=False)
        print(eval_metrics)
        torch.cuda.empty_cache()
        print('Evaluation Done!\n')

    
