import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.stats import gaussian_kde

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def elems_to_edges(elems):
    '''
    elems_to_edges: Takes in an array of triangular mesh elements and 
    outputs an array containing all of their edges

    WARNING -- This function is very simple, but quite slow!
    
    elems - (nt x 3) array of triangular elements, where nt is the number
    of triangles

    Returns:
    - edges, an (ne x 2) array containing all (ne) 1-directional edge pairs
    '''

    rows,_ = np.shape(elems)
    pairs = [(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)]
    
    edges = []
    
    for i in range(rows):
        for pair in pairs:
            edges.append([elems[i,pair[0]], elems[i,pair[1]]])
    
    edges = np.array(edges)
    edges = np.unique(edges,axis=0)
    return edges


def get_edge_lengths(pts,edges):
    starts = pts[edges[:,0],:]
    ends = pts[edges[:,1],:]
    lens = np.linalg.norm(starts - ends,axis=1)
    return lens

def get_kde(graph):
    xy = graph.x[:,:2].detach().numpy()
    kde = gaussian_kde(xy.T, bw_method=0.025)
    return kde

def sample_kde(kde, n, seed = 1):
    xy = kde.resample(n, seed).T
    return xy

def get_dt_edges(xy, thres = 1):
    tri = Delaunay(xy)
    edges = elems_to_edges(tri.simplices)
    lens = get_edge_lengths(xy, edges)
    mean, std = np.mean(lens), np.std(lens)
    ub = mean + thres*std
    edges = [edges[i] for i in range(len(lens)) if lens[i] < ub]
    return np.array(edges)

def get_knn_edges(xy, k=12):
    if type(xy) != torch.Tensor:
        xy = torch.tensor(xy, dtype=torch.float)
    return knn_graph(xy, k=k)


def create_new_graph(xy, edges):
    if type(xy) != torch.Tensor:
        xy = torch.tensor(xy, dtype=torch.float)
    if type(edges) != torch.Tensor:
        edges = torch.tensor(edges, dtype=torch.long)
    if edges.shape[0] != 2:
        edges = edges.T
    return Data(x=xy, edge_index = edges)

def cluster_graph_nodes(fine, coarse):
    xyf = fine.x[:,:2].detach().numpy()
    xyc = coarse.x[:,:2].detach().numpy()
    D = np.square((np.subtract.outer(xyf[:,0],xyc[:,0])))+np.square((np.subtract.outer(xyf[:,1],xyc[:,1])))
    failsafe = np.argmin(D, axis=0) # Make sure each cluster has at least 1 member
    clusters = np.argmin(D, axis=1)
    clusters[failsafe] = np.arange(len(failsafe))
    return clusters

def get_tree_clusters(tree, factor, cluster_list):
    if tree.children <= factor:
        cluster_list.append(tree.idx)
        return
    get_tree_clusters(tree.less, factor, cluster_list)
    get_tree_clusters(tree.greater, factor, cluster_list)
    return cluster_list

def kdtree_cluster(xy, factor = 4, dimension = 3):
    kdt = KDTree(xy, balanced_tree=True, leafsize=factor + 1)
    cluster_list = []
    get_tree_clusters(kdt.tree, factor + 1, cluster_list)
    clusters = np.zeros(xy.shape[0],dtype=np.int64)
    pts = np.zeros((len(cluster_list), dimension))
    for i, idx in enumerate(cluster_list):
        pts[i,:] = np.mean(xy[idx,:],axis=0)
        clusters[idx] = i

    return clusters, pts

class Coarsen:
    def __init__(self, graph, N, factor = 4, node_method="KDTree", edge_method="KNN", z_threshold=1, k=12, dimension = 3, replace_edges=False):
        if edge_method == "DT":
            edgefun = lambda xy: get_dt_edges(xy, z_threshold)
        else: # "KNN"
            edgefun = lambda xy: get_knn_edges(xy, k=k)
        self.N = N
        self.factor = factor
        self.dimension = dimension
        self.data = [graph.clone(),]
        if replace_edges:
            self.data[0].edge_index = edgefun(self.data[0].x[:,:2])
        self.clusters = []

        if node_method == "KDE":
            kde = get_kde(graph)
            num_points = graph.x.shape[0]
        else:
            xy = self.data[0].x.detach().numpy()[:,:self.dimension]
        
        for i in range(N-1):
            if node_method == "KDE":
                num_points = int(num_points/factor)
                xy = sample_kde(kde, num_points)
                edges = edgefun(xy)
                g = create_new_graph(xy, edges)
                self.data.append(g)
                clusters = cluster_graph_nodes(self.data[i],self.data[i+1])
                self.clusters.append(torch.tensor(clusters, dtype=torch.long))
            else:
                clusters, xy = kdtree_cluster(xy, factor, dimension=self.dimension)
                edges = edgefun(xy)
                g = create_new_graph(xy, edges)
                self.data.append(g)
                self.clusters.append(torch.tensor(clusters, dtype=torch.long))

        # For convenience the main graph's variables can be accessed directly
        self.x = self.data[0].x
        #self.sdf = self.data[0].sdf
        self.edge_index = self.data[0].edge_index
        self.y = self.data[0].y

def coarsen_data(data, N = 4, factor = 4, node_method="KDTree", edge_method="KNN", z_threshold=1, k=12, replace_edges=True):
    return Coarsen(data, N=N, factor=factor, node_method=node_method, edge_method=edge_method,
                   z_threshold=z_threshold, k=k, replace_edges=replace_edges)

def coarsen_dataset(dataset, N = 4, factor = 4, node_method="KDTree", edge_method="KNN", z_threshold=1, k=12, replace_edges=True):
    new_dataset = []
    for data in dataset:
        new_dataset.append(Coarsen(data, N=N, factor=factor, node_method=node_method, edge_method=edge_method,
                                   z_threshold=z_threshold, k=k, replace_edges=replace_edges))
    return new_dataset

def merge_coarsened_data(data_list):
    if len(data_list) == 1:
        return data_list[0]
    else:
        data = data_list[0]
        for d in data_list[1:]:
            for j in range(data.N):
                current_node_count = data.data[j].x.shape[0]
                data.data[j].edge_index = torch.cat([data.data[j].edge_index, current_node_count + d.data[j].edge_index],dim=1)
                data.data[j].x = torch.cat([data.data[j].x, d.data[j].x],dim=0)
                if j == 0:
                    data.data[j].y = torch.cat([data.data[j].y, d.data[j].y],dim=0)
                if j < data.N - 1:
                    current_clusters = torch.max(data.clusters[j]) + 1
                    data.clusters[j] = torch.cat([data.clusters[j], current_clusters + d.clusters[j]],dim=0)

        data.x = data.data[0].x
        data.y = data.data[0].y
        data.edge_index = data.data[0].edge_index
        return data