import os
import os.path as osp

import numpy as np

from .data_set import DataSet

# the function is refered 
# GaitSet:https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/data_loader.py
# the load_data function has been moved to "./utils/initialization.py"
def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle=False, cache=True):
    seq_dir = list()
    date = list()
    label = list()
    train_source = None
    test_source = None
    if dataset is 'OUMVLP' or dataset is 'CASIA-B':
        print("Error: Not implemented!!")
    else:
        # load data
        for _label in sorted(list(os.listdir(dataset_path))):
            label_path = osp.join(dataset_path, _label)
            for seqs in sorted(list(os.listdir(label_path))):
                seqs_path = osp.join(label_path, seqs,"normalization")  
                seq_dir.append([seqs_path])
                label.append(_label)
                date.append(seqs[:-3])

        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        if not osp.exists(pid_fname):
            pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)

        pid_list = np.load(pid_fname)
        train_list = pid_list[0]
        test_list = pid_list[1]
        train_source = DataSet(
            [seq_dir[i] for i, l in enumerate(label) if l in train_list],
            [date[i] for i, l in enumerate(label) if l in train_list],
            [label[i] for i, l in enumerate(label) if l in train_list],
            cache, resolution)
        test_source = DataSet(
            [seq_dir[i] for i, l in enumerate(label) if l in test_list],
            [date[i] for i, l in enumerate(label) if l in test_list],
            [label[i] for i, l in enumerate(label) if l in test_list],
            cache, resolution)

    return train_source, test_source
