import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import xarray as xr
import random
import scipy.io as sio
import torch

class PoseDataSet(tordata.Dataset):
    def __init__(self, seq_dir, date, label, config):
        self.seq_dir = seq_dir
        print(seq_dir[0])
        self.date = date
        self.label = label
        self.config = config
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.cache = self.config.data.cache
        self.frame_num = self.config.data.frame_num
        self.label_set = set(self.label)
        self.date_set = set(self.date)
        _ = np.zeros((len(self.label_set),
                      len(self.date_set))).astype('int')
        _ -= 1

        self.index_dict = xr.DataArray(                   
            _,
            coords={'label': sorted(list(self.label_set)),
                    'date': sorted(list(self.date_set))},
            dims=['label', 'date'])

        for i in range(self.data_size):
            _label = self.label[i]
            _date = self.date[i]
            self.index_dict.loc[_label, _date] = i

    def load_all_data(self):
        for i in range(self.data_size):                     # load data process
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)   

    def loader(self, path):
        return self.__loader__(path)

    def __loader__(self, path):
        return sio.loadmat(path[0])['matrix']

    def __getitem__(self, index):
        # pose sequence sampling
        # sample index is from all date
        if not self.cache:
            data = self.__loader__(self.seq_dir[index])

        elif self.data[index] is None:

            data = self.__loader__(self.seq_dir[index])
            self.data[index] = data
        else:
            data = self.data[index]

        # mat_len = data.shape[1]
        # frame_random = random.randint(0, mat_len - self.frame_num - 1)
        # print(self.frame_num)
        # sub_data = np.asarray(data[:, frame_random : frame_random + self.frame_num])
        return data, data.shape[1], self.date[index], self.label[
            index],


    def load_mat(self, filepath):
        mat = sio.loadmat(filepath)['matrix']
        return mat

    def __len__(self):
        return len(self.label)