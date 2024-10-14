import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim

from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as utils

def get_default_conv_info(name):
        if name == "EdgeConv":
            return dict(type=name, dims=[32,32], aggr="max")
        elif name == "GCNConv":
            return dict(type=name, aggr="max")
        elif name == "SAGEConv":
            return dict(type=name, aggr="max", normalize=False)
        elif name == "GATConv":
            return dict(type=name, heads=1, aggr="max", normalize=False)

class GraphUNet(torch.nn.Module):
    def __init__(self, num_layers=2, num_channels=8, mlp_dims = [128, 128], conv_info = None):
        super().__init__()
        if conv_info is None:
            conv_info = dict(type="EdgeConv", dims=[128,128], aggr="max")

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.mlp_dims = mlp_dims
        self.conv_info = conv_info
        self.Lconvs = torch.nn.ModuleList()
        self.Rconvs = torch.nn.ModuleList()
        #self.Lconvs2 = torch.nn.ModuleList()
        #self.Rconvs2 = torch.nn.ModuleList()
        self.Lnorms = torch.nn.ModuleList()
        self.Rnorms = torch.nn.ModuleList()

        for i in range(self.num_layers):
            no = num_channels
            if i == 0:
                ni = 3
            else:
                ni = num_channels + 2

            self.Lconvs.append(self.init_conv(conv_info, ni, no))
            #self.Lconvs2.append(self.init_conv(conv_info, no, no))
            self.Rconvs.append(self.init_conv(conv_info, (no*2 + 2), no))
            #self.Rconvs2.append(self.init_conv(conv_info, no, no))
            self.Lnorms.append(gnn.norm.BatchNorm(no))
            self.Rnorms.append(gnn.norm.BatchNorm(no))

        self.mlp_out = self.get_mlp([no, *mlp_dims, 1])

    def get_mlp(self, dims, activation=nn.ReLU):
        modules = []
        for i, dim in enumerate(dims[:-1]):
            modules.append(nn.Linear(dim, dims[i+1]))
            if i < len(dims) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    def init_conv(self, conv_info, n_in, n_out):
        name = conv_info["type"]
        if name == "EdgeConv":
            return gnn.EdgeConv(self.get_mlp([2*n_in, *conv_info["dims"], n_out]), aggr=conv_info["aggr"])
        elif name == "GCNConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"])
        elif name == "SAGEConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"], normalize=conv_info["normalize"])
        elif name == "GATConv":
            return gnn.GATConv(n_in, n_out, aggr=conv_info["aggr"], heads=conv_info["heads"])

    def forward(self, data, device):
        x = []
        x.append(data.data[0].x.to(device))
        for i in range(self.num_layers - 1):
            self.Lconvs[i].to(device)
            if i > 0:
                x[i] = self.Lconvs[i](torch.cat([data.data[i].x[:,:2].to(device), x[i].to(device)], 1).to(device), data.data[i].edge_index.to(device))
            else:
                x[i] = self.Lconvs[i](x[i].to(device), data.data[i].edge_index.to(device))
                
            #x[i] = torch.relu(x[i])
            #x[i] = self.Lconvs2[i](x[i], data.data[i].edge_index)
            x[i] = torch.relu(x[i])
            x[i] = self.Lnorms[i](x[i].to(device))
            x.append(gnn.avg_pool_x(data.clusters[i].to(device), x[i], torch.zeros_like(data.clusters[i]).to(device))[0])

        i = self.num_layers-1
        X = self.Lconvs[i](torch.cat([data.data[i].x[:,:2].to(device), x[i].to(device)], 1).to(device), data.data[i].edge_index.to(device))
        X = torch.relu(X)
        #X = self.Lconvs2[i](x[i], data.data[i].edge_index)
        #X = torch.relu(x[i])
        X = self.Lnorms[i](X)

        for i in range(self.num_layers,0,-1):
            j = i - 1
            X = torch.cat([data.data[j].x[:,:2].to(device), x[j].to(device), X], 1)
            X = self.Rconvs[j](X, data.data[j].edge_index.to(device))
            #X = torch.relu(X)
            #X = self.Rconvs2[j](X, data.data[j].edge_index)
            X = torch.relu(X)

            X = self.Rnorms[j](X)
            if j > 0:
                X = X[data.clusters[j-1]]

        X = self.mlp_out(X)

        return X

class TAGUNetBlock(torch.nn.Module):
    def __init__(self, n_in=64, n_out=64, num_layers=2, num_channels=64, mlp_dims = [128, 128], conv_info = None, residual = False):
        super().__init__()
        if conv_info is None:
            conv_info = dict(type="EdgeConv", dims=[64,64], aggr="max")

        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.mlp_dims = mlp_dims
        self.conv_info = conv_info
        self.Lconvs = torch.nn.ModuleList()
        self.Rconvs = torch.nn.ModuleList()
        self.Lnorms = torch.nn.ModuleList()
        self.Rnorms = torch.nn.ModuleList()

        for i in range(self.num_layers):
            no = num_channels
            if i == 0:
                ni = n_in
            else:
                ni = num_channels

            self.Lconvs.append(self.init_conv(conv_info, ni, no))
            self.Rconvs.append(self.init_conv(conv_info, no*2, no))
            self.Lnorms.append(gnn.norm.BatchNorm(no))
            self.Rnorms.append(gnn.norm.BatchNorm(no))

        self.mlp_out = self.get_mlp([no, *mlp_dims, n_out])
        self.residual = residual

    def get_mlp(self, dims, activation=nn.ReLU):
        modules = []
        for i, dim in enumerate(dims[:-1]):
            modules.append(nn.Linear(dim, dims[i+1]))
            if i < len(dims) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    def init_conv(self, conv_info, n_in, n_out):
        name = conv_info["type"]
        if name == "EdgeConv":
            return gnn.EdgeConv(self.get_mlp([2*n_in, *conv_info["dims"], n_out]), aggr=conv_info["aggr"])
        elif name == "GCNConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"])
        elif name == "SAGEConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"], normalize=conv_info["normalize"])
        elif name == "GATConv":
            return gnn.GATConv(n_in, n_out, aggr=conv_info["aggr"], heads=conv_info["heads"])

    def unet_step(self, X, edges, clusters, device):
        x = []
        x.append(X.to(device))
        for i in range(self.num_layers):
            x[i] = self.Lconvs[i](x[i], edges[i])
            x[i] = torch.relu(x[i])
            x[i] = self.Lnorms[i](x[i])
            if i < self.num_layers - 1:
                x.append(gnn.avg_pool_x(clusters[i], x[i], torch.zeros_like(clusters[i]).to(device))[0])
            else:
                X = x[i]

        for i in range(self.num_layers,0,-1):
            j = i - 1
            X = torch.cat([x[j], X], 1)
            X = self.Rconvs[j](X, edges[j])
            X = torch.relu(X)
            X = self.Rnorms[j](X)
            if j > 0:
                X = X[clusters[j-1]]

        return X

    def forward(self, X, data, device):
        X = X.to(device)
        clusters = [cluster.to(device) for cluster in data.clusters]
        edges = [data.data[i].edge_index.to(device) for i in range(self.num_layers)]
        
        if self.residual:
            X = X + self.unet_step(X, edges, clusters, device)
            X = X + self.mlp_out(X)
        else:
            X = self.unet_step(X, edges, clusters, device)
            X = self.mlp_out(X)

        return X

class GraphUNetResid(torch.nn.Module):
    def __init__(self, n_in=64, n_out=64, num_layers=2, num_channels=64, mlp_dims = [128, 128], conv_info = None):
        super().__init__()
        if conv_info is None:
            conv_info = dict(type="EdgeConv", dims=[64,64], aggr="max")

        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.mlp_dims = mlp_dims
        self.conv_info = conv_info
        self.Lconvs = torch.nn.ModuleList()
        self.Rconvs = torch.nn.ModuleList()
        self.Lnorms = torch.nn.ModuleList()
        self.Rnorms = torch.nn.ModuleList()

        for i in range(self.num_layers):
            no = num_channels
            if i == 0:
                ni = n_in
            else:
                ni = num_channels

            self.Lconvs.append(self.init_conv(conv_info, ni, no))
            self.Rconvs.append(self.init_conv(conv_info, no*2, no))
            self.Lnorms.append(gnn.norm.BatchNorm(no))
            self.Rnorms.append(gnn.norm.BatchNorm(no))

        self.mlp_out = self.get_mlp([no, *mlp_dims, n_out])

    def get_mlp(self, dims, activation=nn.ReLU):
        modules = []
        for i, dim in enumerate(dims[:-1]):
            modules.append(nn.Linear(dim, dims[i+1]))
            if i < len(dims) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    def init_conv(self, conv_info, n_in, n_out):
        name = conv_info["type"]
        if name == "EdgeConv":
            return gnn.EdgeConv(self.get_mlp([2*n_in, *conv_info["dims"], n_out]), aggr=conv_info["aggr"])
        elif name == "GCNConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"])
        elif name == "SAGEConv":
            return gnn.GCNConv(n_in, n_out, aggr=conv_info["aggr"], normalize=conv_info["normalize"])
        elif name == "GATConv":
            return gnn.GATConv(n_in, n_out, aggr=conv_info["aggr"], heads=conv_info["heads"])

    def unet_step(self, X, edges, clusters, device):
        x = []
        x.append(X.to(device))
        for i in range(self.num_layers):
            x[i] = self.Lconvs[i](x[i], edges[i])
            x[i] = torch.relu(x[i])
            x[i] = self.Lnorms[i](x[i])
            if i < self.num_layers - 1:
                x.append(gnn.avg_pool_x(clusters[i], x[i], torch.zeros_like(clusters[i]).to(device))[0])
            else:
                X = x[i]

        for i in range(self.num_layers,0,-1):
            j = i - 1
            X = torch.cat([x[j], X], 1)
            X = self.Rconvs[j](X, edges[j])
            X = torch.relu(X)
            X = self.Rnorms[j](X)
            if j > 0:
                X = X[clusters[j-1]]

        return X

    def forward(self, X, data, device):
        X = X.to(device)
        clusters = [cluster.to(device) for cluster in data.clusters]
        edges = [data.data[i].edge_index.to(device) for i in range(self.num_layers)]

        X = X + self.unet_step(X, edges, clusters, device)

        X = X + self.mlp_out(X)

        return X
    
class MultiTAGUNet(torch.nn.Module):
    def __init__(self, n_in, n_out, num_layers=2, num_channels=64, depth = 2, mlp_dims = [128, 128], conv_info = None):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.mlp_dims = mlp_dims
        self.conv_info = conv_info
        self.depth = depth

        self.enc = nn.Linear(n_in, num_channels)
        self.dec = nn.Linear(num_channels, n_out)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphUNetResid(n_in=num_channels, n_out=num_channels, num_channels=num_channels, mlp_dims=mlp_dims, conv_info=conv_info)
            self.layers.append(layer)

    def forward(self, data, device):
        X = self.enc(data.x.to(device))
        for layer in self.layers:
            X = layer(X, data, device)
        X = self.dec(X)
        return X

        






class JustConvNet(torch.nn.Module):
    def __init__(self, num_convs=2, conv_dims=[32,32], channels=8, mlp_dims = [128, 128]):
        super().__init__()

        convs = []
        for _ in range(num_convs):
            conv = gnn.EdgeConv(self.get_mlp([2*channels, *conv_dims, channels]), aggr="max")
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

        self.enc = nn.Linear(3,channels)
        self.mlp = self.get_mlp([channels,*mlp_dims,1])

    def get_mlp(self, dims, activation=nn.ReLU):
        modules = []
        for i, dim in enumerate(dims[:-1]):
            modules.append(nn.Linear(dim, dims[i+1]))
            if i < len(dims) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    def forward(self, data, device):
        X = self.enc(data.x.to(device))
        edge_index = data.edge_index.to(device)
        for module in self.convs:
            X = module(x=X, edge_index=edge_index)
            X = F.relu(X)
        X = self.mlp(X)
        return X