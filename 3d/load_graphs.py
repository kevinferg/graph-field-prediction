import torch
from torch_geometric.data import Data
from parse_netfabb_results import *
import os
from torch.utils.data import Dataset, DataLoader
from coarsen import *

def cube_elems_to_edges(elems):
    N, _ = np.shape(elems)
    # Assuming node indexing as in: https://www.strand7.com/strand7r3help/Content/Topics/Elements/ElementsBricksElementTypes.htm
    edge_A = np.array([1,2,3,4,5,6,7,8,1,4,2,3])-1
    edge_B = np.array([2,3,4,1,6,7,8,5,5,8,6,7])-1
    edges = np.zeros([2,N*12])
    for i in range(N):
        new_idx = np.arange(i*12, (i+1)*12)
        edges[0, new_idx] = elems[i, edge_A]
        edges[1, new_idx] = elems[i, edge_B]

    edges = np.sort(edges, axis=0)
    edges = np.unique(edges, axis=1)

    edges = np.concatenate([edges, np.flipud(edges)], axis=1)
    return edges

def get_displacement_graph(name, npz_mode=False):
    if npz_mode:
        results = np.load(name)
        verts, elems, disp = results["verts"], results["elems"], results["disp"]
    else:
        verts, elems, disp = get_displacement_results_only(name)
    edges = cube_elems_to_edges(elems)

    x = torch.tensor(verts, dtype=torch.float)
    y = torch.tensor(disp[:,2].flatten(), dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


class NetFabbDataset(Dataset):
    def __init__(self, path, npz_mode=False, **kwargs):
        self.path = path
        self.npz_mode = npz_mode
        self.options = kwargs
        
        print(f"Loading displacement results from: {path}")
        files = os.listdir(path)
        if npz_mode:
            self.files = files
            print(f"Done. Using all {len(self.files)} .npz files")
            return

        self.files = []
        for file in files:
            results = get_file_info(os.path.join(self.path, file))
            if 1 == len(results):  
                error = results["error"]
                print(f"{file:<20} -- {error}")
            else:
                self.files.append(file)
        print(f"Loaded {len(self.files)} parts successfully (of {len(files)})")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = get_displacement_graph(os.path.join(self.path,self.files[idx]), self.npz_mode)
        return coarsen_data(X, **self.options)
    
def split_dataset(dataset, train_frac=0.8, seed=216):
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator)
    print(f"Splitting dataset with seed {seed}: {train_size} training, {test_size} testing")
    return train_dataset, test_dataset
    
class NetFabbDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=merge_coarsened_data, **kwargs)