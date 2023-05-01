import numpy as np
import torch
import torch_geometric.nn
import torch_scatter
from itertools import chain
from torch_geometric.nn import MetaLayer
import os



# ================== Encoder ================== #
class NodeEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, node_state):
        out = self.model(node_state)

        return out


class EdgeEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, edge_properties):
        out = self.model(edge_properties)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, node_input_size, edge_input_size, hidden_size=128, output_size=128):
        super(Encoder, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.node_encoder = NodeEncoder(self.node_input_size, self.hidden_size, self.output_size)
        self.edge_encoder = EdgeEncoder(self.edge_input_size, self.hidden_size, self.output_size)

    def forward(self, node_states, edge_properties):
        node_embedding = self.node_encoder(node_states)
        edge_embedding = self.edge_encoder(edge_properties)

        return node_embedding, edge_embedding


# ================== Processor ================== #

class EdgeModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # u_expanded = u.expand([src.size()[0], -1])
        # model_input = torch.cat([src, dest, edge_attr, u_expanded], 1)
        # out = self.model(model_input)
        model_input = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.model(model_input)
        return out


class NodeModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        _, edge_dst = edge_index
        edge_attr_aggregated = torch_scatter.scatter_add(edge_attr, edge_dst, dim=0, dim_size=x.size(0))
        model_input = torch.cat([x, edge_attr_aggregated, u[batch]], dim=1)
        out = self.model(model_input)

        return out


class GlobalModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(GlobalModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_attr_mean = torch_scatter.scatter_mean(x, batch, dim=0)
        edge_attr_mean = torch_scatter.scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        # print("x shape: ", x.shape)
        # print("edge_index shape: ", edge_index.shape)
        # print("edge_attr shape: ", edge_attr.shape)
        # print("batch shape: ", batch.shape)
        # print("u shape: ", u.shape)
        # print("batch: ", batch)
        # print("node_attr_mean shape: ", node_attr_mean.shape)
        # print("edge_attr_mean shape: ", edge_attr_mean.shape, flush=True)
        model_input = torch.cat([u, node_attr_mean, edge_attr_mean], dim=1)
        out = self.model(model_input)
        assert out.shape == u.shape
        return out


class GNBlock(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128):

        super(GNBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if use_global:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   GlobalModel(self.input_size[2], self.hidden_size, global_size))
        else:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   None)

    def forward(self, x, edge_index, edge_attr, u, batch):

        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        x, edge_attr, u = self.model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


class Processor(torch.nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128, layers=10):
        super(Processor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_global = use_global
        self.global_size = global_size
        self.gns = torch.nn.ModuleList([
            GNBlock(self.input_size, self.hidden_size, self.output_size, self.use_global, global_size=global_size)
            for _ in range(layers)])

    def forward(self, x, edge_index, edge_attr, u, batch):
        # def forward(self, data):
        # x, edge_index, edge_attr, u, batch = data.node_embedding, data.neighbors, data.edge_embedding, data.global_feat, data.batch
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if len(u.shape) == 1:
            u = u[None]

        x_res, edge_attr_res, u_res = x, edge_attr, u  # Don't do it! In-place residual is problematic
        for gn in self.gns:
            x_new, edge_attr_new, u_new = gn(x_res, edge_index, edge_attr_res, u_res, batch)
            x_res = x_new + x_res
            edge_attr_res = edge_attr_new + edge_attr_res
            u_res = u_new + u_res
        return x_new, edge_attr_new, u_new


# ================== Decoder ================== #

class Decoder(torch.nn.Module):

    def __init__(self, input_size=128, hidden_size=128, output_size=3):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, node_feat, res=None):
        out = self.model(node_feat)
        if res is not None:
            out += res

        return out


class GNNModel(torch.nn.Module):
    def __init__(self, args, proc_layer_num, global_size, output_size=1):
        super(GNNModel, self).__init__()
        self.args = args
        self.use_global = True if global_size > 1 else False
        self.dyn_models = torch.nn.ModuleDict({'encoder': Encoder(args.node_dim, args.edge_dim),
                                               'processor': Processor(
                                                   [3 * 128 + global_size, 2 * 128 + global_size, 2 * 128 + global_size]
                                                   , use_global=self.use_global,
                                                   layers=proc_layer_num,
                                                   global_size=global_size),
                                               'decoder': Decoder(output_size=output_size)})

    def forward(self, data):
        '''
        edge_index,
        x,
        edge_attr,
        gt_rwd,
        gt_accel,
        x_batch
        '''
        out = {}
        node_embedding, edge_embedding = self.dyn_models['encoder'](data['x'], data['edge_attr'])

        n_nxt, e_nxt, lat_nxt = self.dyn_models['processor'](node_embedding,
                                                               data['edge_index'],
                                                               edge_embedding,
                                                               u=data['u'],
                                                               batch=data['batch'])
        out['target'] = self.dyn_models['decoder'](n_nxt)
        out['n_nxt'] = n_nxt[data['node_mesh_mapping']] if 'node_mesh_mapping' in data else n_nxt
        out['lat_nxt'] = lat_nxt
        return out

    def load_model(self, model_dir, input_type, epoch, load_names='all', load_optim=False, optim=None):
        if epoch > 0:
            model_path = model_dir + '/{}_dyn_{}.pth'.format(input_type, epoch)
        else:
            model_path = model_dir + '/{}_dyn_best.pth'.format(input_type)
        ckpt = torch.load(model_path)
        optim_path = model_path.replace('dyn', 'optim')
        if load_names == 'all':
            for k, v in self.dyn_models.items():
                self.dyn_models[k].load_state_dict(ckpt[k])
            print('Loaded saved ckp from {}'.format(model_path))
        else:
            for model_name in load_names:
                self.dyn_models[model_name].load_state_dict(ckpt[model_name])
            print('Loaded saved ckp from {}'.format(model_path), load_names)
        if load_optim and os.path.exists(optim_path):
            optim.load_state_dict(torch.load(optim_path))

    def save_model(self, root_path, m_name, suffix, optim):
        """
        Regular saving: {input_type}_dyn_{epoch}.pth
        Best model: {input_type}_dyn_best.pth
        Optim: {input_type}_optim_{epoch}.pth
        """
        model_path = os.path.join(root_path, '{}_{}_{}.pth'.format(m_name, 'dyn', suffix))
        torch.save({k: v.state_dict() for k, v in self.dyn_models.items()}, model_path)
        optim_path = os.path.join(root_path, '{}_{}_{}.pth'.format(m_name, 'optim', suffix))
        torch.save(optim.state_dict(), optim_path)

    def set_mode(self, mode='train'):
        assert mode in ['train', 'eval', 'valid']
        for model in self.dyn_models.values():
            if mode == 'eval' or mode == 'valid':
                model.eval()
            else:
                model.train()

    def param(self):
        model_parameters = list(chain(*[list(m.parameters()) for m in self.dyn_models.values()]))
        return model_parameters

    def to(self, device):
        for model in self.dyn_models.values():
            model.to(device)

    def freeze(self, tgts=None):
        if tgts is None:
            for m in self.dyn_models.values():
                for para in m.parameters():
                    para.requires_grad = False
        else:
            for tgt in tgts:
                m = self.dyn_models[tgt]
                for para in m.parameters():
                    para.requires_grad = False

    def unfreeze(self, tgts=None):
        if tgts is None:
            for m in self.dyn_models.values():
                for para in m.parameters():
                    para.requires_grad = True
        else:
            for tgt in tgts:
                m = self.dyn_models[tgt]
                for para in m.parameters():
                    para.requires_grad = True
