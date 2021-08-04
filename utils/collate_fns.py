import math
import torch
import random
import numpy as np
from utils.random import random_clip, random_select, randint

class Collate_fn_select(object):
    def __init__(self, frame_num, sample_type):
        self.batch = None
        self.frame_num = frame_num
        self.step = max(int(0.4*(self.frame_num)),1)
        self.sample_type = sample_type

    def __call__(self, batch):
        self.batch = batch
        # for the training set, sample type is random
        batch_size = len(self.batch)
        feature_num = len(self.batch[0][0])
        seqs = [self.batch[i][0] for i in range(batch_size)]
        frame_sets = [self.batch[i][1] for i in range(batch_size)]
        date = [self.batch[i][2] for i in range(batch_size)]
        label = [self.batch[i][3] for i in range(batch_size)]

        batch = [seqs, date, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random_select(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            batch = [*seqs, date, label, None]
        else:
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]

            seqs_temp = list()
            date_temp = list()
            label_temp = list()

            real_step = min(self.frame_num, self.step)

            for i in range(batch_size):
                mat_data = seqs[0][i]  # wheath exiting [0]?
                mat_len = len(frame_sets[i])

                mini_batch_num = int((mat_len - self.frame_num + 1) / real_step)

                if mini_batch_num < 2:  #

                    mat_data_temp = None
                    if mat_len < self.frame_num:
                        # padding
                        mat_data_temp = np.pad(mat_data, ((0, self.frame_num - mat_len), (0, 0), (0, 0)), 'constant', constant_values = 0)
                    else:
                        mat_data_temp = mat_data[0: self.frame_num, :, :]

                    # mat_data_temp = mat_data_temp[np.newaxis, :, :, :]
                    seqs_temp.append(mat_data_temp)
                    date_temp.append(date[i])
                    label_temp.append(label[i])

                else:

                    for j in range(mini_batch_num):
                        mat_data_temp = mat_data[j * real_step: j * real_step + self.frame_num, :, :]

                        seqs_temp.append(mat_data_temp)
                        date_temp.append(date[i])
                        label_temp.append(label[i])
            seqs_temp = np.asarray(seqs_temp)
            # print(seqs_temp.shape)
            batch = [seqs_temp, date_temp, label_temp, None]
        return batch


class Collate_fn_clip_probe(object):
    def __init__(self, frame_num, sample_type):
        self.batch = None
        self.frame_num = frame_num
        self.sample_type = sample_type
        self.step = max(int(0.4 * (self.frame_num)), 1)

    def __call__(self, batch):
        self.batch = batch
        # for the training set, sample type is random
        batch_size = len(self.batch)
        feature_num = len(self.batch[0][0])
        seqs = [self.batch[i][0] for i in range(batch_size)]
        frame_sets = [self.batch[i][1] for i in range(batch_size)]
        date = [self.batch[i][2] for i in range(batch_size)]
        label = [self.batch[i][3] for i in range(batch_size)]
        batch = [seqs, date, label, None]

        def select_frame(index):
            sample = seqs[index]
            _ = np.asarray([feature.values for feature in sample])
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        batch = [*seqs, date, label, None]
        return batch
 
class Collate_fn_clip_OUMVLP(object):
    def __init__(self, frame_num, sample_type):
        self.batch = None
        self.frame_num = frame_num
        self.sample_type = sample_type
        self.step = max(int(0.4 * (self.frame_num)), 1)

    def __call__(self, batch):
        # getitem return data, frame_set, self.view[index], self.seq_type[index], self.label[index]
        self.batch = batch
        # for the training set, sample type is random
        batch_size = len(self.batch)
        feature_num = len(self.batch[0][0])
        seqs = [self.batch[i][0] for i in range(batch_size)]
        frame_sets = [self.batch[i][1] for i in range(batch_size)]
        view = [self.batch[i][2] for i in range(batch_size)]
        seq_type = [self.batch[i][3] for i in range(batch_size)]
        label = [self.batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random_clip(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            batch = [*seqs, view, seq_type, label, None]
        else:
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]

            seqs_temp = list()
            view_temp = list()
            seq_type_temp = list()
            label_temp = list()
            real_step = min(self.frame_num, self.step)

            for i in range(batch_size):
                mat_data = seqs[0][i]  # wheath exiting [0]?
                mat_len = len(frame_sets[i])

                mini_batch_num = int((mat_len - self.frame_num + 1) / real_step)

                if mini_batch_num < 2:  #

                    mat_data_temp = None
                    if mat_len < self.frame_num:
                        # padding
                        mat_data_temp = np.pad(mat_data, ((0, self.frame_num - mat_len), (0, 0), (0, 0)), 'constant',
                                               constant_values=0)
                    else:
                        mat_data_temp = mat_data[0: self.frame_num, :, :]

                    seqs_temp.append(mat_data_temp)
                    view_temp.append(view[i])
                    seq_type_temp.append(seq_type[i])
                    label_temp.append(label[i])
                else:

                    for j in range(mini_batch_num):
                        mat_data_temp = mat_data[j * real_step: j * real_step + self.frame_num, :, :]

                        seqs_temp.append(mat_data_temp)
                        view_temp.append(view[i])
                        seq_type_temp.append(seq_type[i])
                        label_temp.append(label[i])
            seqs_temp = np.asarray(seqs_temp)

            batch = [seqs_temp, view_temp, seq_type_temp, label_temp, None]

        return batch

class Collate_fn_clip(object):
    def __init__(self, frame_num, sample_type):
        self.batch = None
        self.frame_num = frame_num
        self.sample_type = sample_type
        self.step = max(int(0.4*(self.frame_num)),1)


    def __call__(self, batch):
        self.batch = batch
        # for the training set, sample type is random
        batch_size = len(self.batch)
        feature_num = len(self.batch[0][0])
        seqs = [self.batch[i][0] for i in range(batch_size)]
        frame_sets = [self.batch[i][1] for i in range(batch_size)]
        date = [self.batch[i][2] for i in range(batch_size)]
        label = [self.batch[i][3] for i in range(batch_size)]
        batch = [seqs, date, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random_clip(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            batch = [*seqs, date, label, None]
        else:
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            
            seqs_temp = list()
            date_temp = list()
            label_temp = list()
            real_step = min(self.frame_num, self.step)
            
            for i in range(batch_size):
                mat_data = seqs[0][i]       # wheath exiting [0]?
                mat_len = len(frame_sets[i])

                mini_batch_num = int((mat_len - self.frame_num + 1)/real_step)

                if mini_batch_num < 2:                       #

                    mat_data_temp = None
                    if mat_len < self.frame_num:
                        # padding
                        mat_data_temp = np.pad(mat_data, ((0, self.frame_num - mat_len), (0, 0), (0, 0)), 'constant', constant_values = 0)
                    else:
                        mat_data_temp = mat_data[0 : self.frame_num, :, : ]
                    # mat_data_temp = mat_data_temp[np.newaxis, :, :, :]
                    seqs_temp.append(mat_data_temp)
                    date_temp.append(date[i])
                    label_temp.append(label[i])
                else:

                    for j in range(mini_batch_num):
                        
                        mat_data_temp = mat_data[j * real_step : j*real_step + self.frame_num, :, : ]
                        # mat_data_temp = mat_data_temp[ :, :, :]

                        seqs_temp.append(mat_data_temp)
                        date_temp.append(date[i])
                        label_temp.append(label[i])
            seqs_temp = np.asarray(seqs_temp)
            # print(seqs_temp.shape)

            batch = [seqs_temp, date_temp, label_temp, None]
        return batch


def update_gallery(data_gallery, date_gallery, label_gallery, frame_num = 32, overlap_per = 0.4):
    step = max(int(overlap_per * (frame_num)), 1)
    real_step = min(frame_num, step)
    data_gallery_temp = list()
    date_gallery_temp = list()
    label_gallery_temp = list()

    len_gallery = len(label_gallery)

    for i in range(len_gallery):
        data_gallery_data = np.asarray(data_gallery[i].values)
        date_gallery_data = date_gallery[i]
        label_gallery_data = label_gallery[i]

        frame_i = data_gallery_data.shape[0]

        mini_batch_num = int((frame_i - frame_num + 1) / real_step)

        if mini_batch_num < 2:  #
            data_gallery_temp.append(data_gallery_data)
            date_gallery_temp.append(date_gallery_data)
            label_gallery_temp.append(label_gallery_data)
        else:

            for j in range(mini_batch_num):
                mat_data_temp = data_gallery_data[j * real_step: j * real_step + frame_num, :, :]
                # mat_data_temp = mat_data_temp[ :, :, :]

                data_gallery_temp.append(mat_data_temp)
                date_gallery_temp.append(date_gallery_data)
                label_gallery_temp.append(label_gallery_data)

    return np.asarray(data_gallery_temp), date_gallery_temp, label_gallery_temp


class Collate_fn_pose(object):
    def __init__(self, frame_num, sample_type):
        self.batch = None
        self.frame_num = frame_num
        self.sample_type = sample_type
        self.step = int(0.4*(self.frame_num))

    def __call__(self, batch):
        # sub_data, self.frame_num, self.date[index], self.label[index],
        self.batch = batch
        # for the training set, sample type is random
        batch_size = len(self.batch)
        feature_num = len(self.batch[0][0])
        seqs = [self.batch[i][0] for i in range(batch_size)]
        frame_sets = [self.batch[i][1] for i in range(batch_size)]
        date = [self.batch[i][2] for i in range(batch_size)]
        label = [self.batch[i][3] for i in range(batch_size)]

        batch = [seqs, date, label, None]

        def select_frame(index):
            data = seqs[index]
            mat_len = frame_sets[index]

            if mat_len < self.frame_num:
                sub_data = np.zeros((36, self.frame_num))
                sub_data[:, 0:mat_len] = data[:, :]

            elif self.frame_num+2 < mat_len:

                frame_random = randint(0, mat_len - self.frame_num - 1)
                # print(self.frame_num)
                sub_data = np.asarray(data[:, frame_random: frame_random + self.frame_num])
            else:
                sub_data = np.zeros((36, self.frame_num))
                sub_data[:, 0:self.frame_num] = data[:, 0:self.frame_num]
            # print(sub_data.shape)
            return sub_data
        if self.sample_type == 'random':
            seqs = list(map(select_frame, range(len(seqs))))
            batch[0] = seqs
        else:
            seqs_temp = list()
            date_temp = list()
            label_temp = list()
            real_step = min(self.frame_num, self.step)

            for i in range(batch_size):
                mat_data = seqs[i]  # wheath exiting [0]?
                mat_len = frame_sets[i]

                mini_batch_num = int((mat_len - self.frame_num + 1) / real_step)

                if mini_batch_num < 2:  #

                    mat_data_temp = None
                    if mat_len < self.frame_num:

                        mat_data_temp = np.pad(mat_data, ((0, 0), (0, self.frame_num - mat_len)), 'constant', constant_values = 0)
                    else:

                        mat_data_temp = mat_data[:, 0: self.frame_num ]

                    # mat_data_temp = mat_data_temp[np.newaxis, :, :, :]
                    seqs_temp.append(mat_data_temp)
                    date_temp.append(date[i])
                    label_temp.append(label[i])

                else:
                    for j in range(mini_batch_num):
                        mat_data_temp = mat_data[ :, j * real_step: j * real_step + self.frame_num ]
                        # mat_data_temp = mat_data_temp[ :, :, :]

                        seqs_temp.append(mat_data_temp)
                        date_temp.append(date[i])
                        label_temp.append(label[i])
            seqs_temp = np.asarray(seqs_temp)
            # print(seqs_temp.shape)
            batch = [seqs_temp, date_temp, label_temp, None]

        return batch


def collate_fn_select(frame_num, sample_type):
    # for the training set, sample type is random
    return Collate_fn_select(frame_num, sample_type)

def collate_fn_clip_OUMVLP(frame_num, sample_type):
    return Collate_fn_clip_OUMVLP(frame_num, sample_type)

def collate_fn_clip(frame_num, sample_type):
    return Collate_fn_clip(frame_num, sample_type)

def collate_fn_clip_probe(frame_num, sample_type):
    return Collate_fn_clip_probe(frame_num, sample_type)

def collate_fn_pose(frame_num, sample_type):  # for pose data, the collate_fn is default.
    # wheather adding some other 
    return Collate_fn_pose(frame_num, sample_type)

def get_collate_fn(config, frame_num, sample_type):
    func = globals().get('collate_fn_'+config.data.collate_fn)

    return func(frame_num, sample_type)