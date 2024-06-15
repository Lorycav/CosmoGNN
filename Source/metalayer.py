# +
#Graph Neural Network architecture
# -

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList, GELU, Softplus
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from Source.constants import *
from torch_geometric.utils import scatter

# Model for updating edge attritbutes

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=True):
        super().__init__()

        # for residual network
        self.residuals = residuals
        # for batch normalization
        self.norm = norm

        layers = [
            Linear(node_in*2 + edge_in, hid_channels), 
            GELU(),
            Linear(hid_channels, edge_out)
            ]

        # batch norm
        if self.norm:  layers.append(LayerNorm(edge_out))

        # defining torch model
        self.edge_mlp = Sequential(*layers)

    def forward(self, src, dest, edge_attr):
        # src: edge start
        # dest: edge end
        # src, dest: tensors of shape [E, F_x], where E is the number of edges
        # edge_attr: [E, F_e]

        # building a unique tensor
        # the resulting tensor out will have the shape [E, 2*F_x + F_e]
        out = torch.cat([src, dest, edge_attr], dim=1)

        # forward pass
        out = self.edge_mlp(out)

        # when we need resnet
        if self.residuals:
            out = out + edge_attr

        return out



# Model for updating node attritbutes

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_out, hid_channels, residuals=True, norm=True):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [
            Linear(node_in + 3*edge_out + 1, hid_channels),
            GELU(),
            Linear(hid_channels, node_out)
            ]

        # batch normalization
        if self.norm:  layers.append(LayerNorm(node_out))

        # building torch model
        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x = node features
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter(out, col, dim=0, dim_size=x.size(0), reduce="sum")
        out2 = scatter(out, col, dim=0, dim_size=x.size(0), reduce="max")
        out3 = scatter(out, col, dim=0, dim_size=x.size(0), reduce="mean")
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)

        # when we need resnet
        if self.residuals:
            out = out + x

        return out
    

# Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# Employing the MetaLayer implementation in Pytorch-Geometric

class GNN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out, residuals=True):
        super().__init__()

        self.n_layers = n_layers
        self.link_r = linkradius
        self.dim_out = dim_out

        # Number of input node features 
        node_in = node_features
        
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        layers = []

        # Encoder graph block
        inlayer = MetaLayer(
            node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False),
            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False)
            )

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph blocks
        for _ in range(n_layers-1):

            lay = MetaLayer(
                node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals)
                )

            layers.append(lay)

        # holding submodules in a list
        self.layers = ModuleList(layers)

        # Final aggregation layer 
        self.outlayer = Sequential(
            Linear(3*node_out+1, hid_channels),
            GELU(),
            Linear(hid_channels, hid_channels),
            GELU(),
            Linear(hid_channels, hid_channels),
            GELU(),
            Linear(hid_channels, self.dim_out),
            Softplus()
            )

    def forward(self, data):

        # Retrieving data
        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        # Multipooling layer
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        out = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Final linear layer
        alpha = 0.85     # variance factor to help training
        out = self.outlayer(out)
        out[:, 1] = alpha * out[:, 1]
    
        return out

