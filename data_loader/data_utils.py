from utils.math_utils import z_score
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, data, n_his):
        self.__data = data
        self.n_his = n_his

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx, :self.n_his, :, :], self.__data[idx, self.n_his:, :, :]


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, c_0=1):
    """
    Gain dataset from the original time series.
    :param len_seq: int, length of the sequence.
    :param data_seq: np.ndarray, [len_seq, n_route * C_0].
    :param offset: start point to make the new dataset.
    :param n_frame: int, n_his + n_pred.
    :param n_route: int, number of the vertices on the graph.
    :param day_slot: int, the number of time slots per day, controlled
                     by the time window(5m as default
    :param c_0: number of the channels of source data.
    :return: np.ndarray, [n_slot, n_frame, n_route, C_0].
    """
    n_slot = day_slot - n_frame + 1
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, c_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i+offset) * day_slot +j
            end = sta + n_frame
            tmp_seq[i * n_slot +j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, c_0])
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame, device, day_slot=288):
    """Generate datasets for training, validation, and test.
    :param file_path: str， the path of the file.
    :param data_config: tuple, the portion of each set.
    :param n_route: int, number of the vertices on the graph.
    :param n_frame: n_his + n_pred.
    :param device: cuda or cpu
    :return:  dict that contains training, validation and test data，stats.
    """

    n_train, n_val, n_test = data_config

    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        raise FileNotFoundError(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the training dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: tensor, [len_seq, n_frame, n_route, C_0].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    x_train = torch.from_numpy(x_train).type(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).type(torch.float32).to(device)
    x_test = torch.from_numpy(x_test).type(torch.float32).to(device)
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    return x_data, x_stats


def loader_gen(data_file, n_train, n_val, n_test, n_his, n_pred, n_route, batch_size, device):
    """  Wrap the dataset with data loaders.
    :param data_file:str, the path of the file
    :param n_train: int, number of days for training datas
    :param n_val: int, number of days for validation datas
    :param n_test: int, number of days for testing datas
    :param n_his: int, length of source series.
    :param n_pred: int, length of target series.
    :param n_route: int, number of routes
    :param batch_size: int, size of each batch
    :param device: cuda or cpu
    :return: dict of dataloaders for training and validation, dict of sizes for training dataset
    validation dataset, dataset for testing, statics for the dataset.
    data shape [batch_size, seq_len, n_route, C_0].
    """

    data_traffic, stats = data_gen(data_file, (n_train, n_val, n_test), n_route, n_his + n_pred, device)
    trainset = TrafficDataset(data_traffic['train'], n_his)
    validset = TrafficDataset(data_traffic['val'], n_his)
    testset = TrafficDataset(data_traffic['test'], n_his)
    train_data_gen = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valid_data_gen = DataLoader(validset, batch_size=batch_size)
    test_data_gen = DataLoader(testset, batch_size=batch_size)
    dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
    dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}
    return dataloaders, dataset_sizes, test_data_gen, stats
