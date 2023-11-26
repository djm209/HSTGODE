import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from tqdm import tqdm
from fastdtw import fastdtw
import csv

files = {
    'XiAn_City': ['xian/original_data.npz', 'xian/distance.csv'],
    'XiAn_City_Cluster': ['xian/original_data_cluster.npz', 'xian/distance_cluster.csv'],
    'pems03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/distance.csv'],
    'pems07': ['PEMS07/pems07.npz', 'PEMS07/distance.csv'],
    'pems08': ['PEMS08/pems08.npz', 'PEMS08/distance.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
    
class DataLoader_cluster(object):
    def __init__(self, xs, ys,xc,yc, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            num_padding = (batch_size - (len(xc) % batch_size)) % batch_size
            x_padding = np.repeat(xc[-1:], num_padding, axis=0)
            y_padding = np.repeat(yc[-1:], num_padding, axis=0)
            xc = np.concatenate([xc, x_padding], axis=0)
            yc = np.concatenate([yc, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, xc, yc = self.xs[permutation], self.ys[permutation], self.xc[permutation], self.yc[permutation]
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_c = self.xc[start_ind: end_ind, ...]
                y_c = self.yc[start_ind: end_ind, ...]
                yield (x_i, y_i, x_c, y_c)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def load_mtogde_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj

def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    #for category in ['train', 'val', 'test']:
        #data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def load_dataset_cluster(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train_cluster', 'val_cluster', 'test_cluster']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler_cluster = StandardScaler(mean=data['x_train_cluster'][..., 0].mean(), std=data['x_train_cluster'][..., 0].std())
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['train_loader_cluster'] = DataLoader_cluster(data['x_train'], data['y_train'],data['x_train_cluster'], data['y_train_cluster'], batch_size)
    data['val_loader_cluster'] = DataLoader_cluster(data['x_val'], data['y_val'],data['x_val_cluster'], data['y_val_cluster'], valid_batch_size)
    data['test_loader_cluster'] = DataLoader_cluster(data['x_test'], data['y_test'],data['x_test_cluster'], data['y_test_cluster'], test_batch_size)
    data['scaler'] = scaler
    data['scaler_cluster'] = scaler_cluster
    return data

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    file = files[filename]
    filepath = "data_generation/"
    data = np.load(filepath + file[0])['data'].transpose((1,0,2))

    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    if not os.path.exists(f'data_generation/data/XiAn_City/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                            axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data_generation/data/XiAn_City/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'data_generation/data/XiAn_City/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')

    # use continuous spatial matrix
    if not os.path.exists(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy', dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres2] = 0
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0) / 2 / num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0) / 2 / num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix

def read_cluster_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """

    filename = args.cluster_filename
    file = files[filename]
    filepath = "data_generation/"
    data = np.load(filepath + file[0])['data'].transpose((1, 0, 2))

    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[1]
    std_value = std_value.reshape(-1)[1]

    if not os.path.exists(f'data_generation/data/XiAn_City/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 1][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                            axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'data_generation/data/XiAn_City/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')

    # use continuous spatial matrix
    if not os.path.exists(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy', dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(f'data_generation/data/XiAn_City/{filename}_spatial_distance.npy')
    # normalization
    # std = np.std(dist_matrix[dist_matrix != float('inf')])
    # mean = np.mean(dist_matrix[dist_matrix != float('inf')])
    # dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres2] = 0
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0) / 2 / num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0) / 2 / num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


