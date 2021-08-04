# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
from os import path as osp
from .time_split import big_day, small_day # for time span exp

import numpy as np
import pickle
from datasets.dataset_factory import get_dataset
import random
import traceback


import pickle
import cv2
import xarray as xr
import pandas as pd
import scipy.io as sio

def load_data_from_path_pose(path):
    mat = sio.loadmat(path)['matrix']
    return mat

def load_data_from_path(path, resolution = 64):
    cut_padding = int(float(resolution) / 64 * 10)
    return img2xarray(path[0], resolution)[:, :, cut_padding:-cut_padding].astype('float32') / 255.0

def img2xarray(filepath, resolution):
    # print(flie_path)
    # load data from a given every file path
    imgs = sorted(list(os.listdir(filepath)))
    frame_list = [np.reshape(
        cv2.resize(cv2.imread(osp.join(filepath, _img_path)), (64, 64), interpolation=cv2.INTER_CUBIC),
        [resolution, resolution, -1])[:, :, 0]
                  for _img_path in imgs
                  if osp.isfile(osp.join(filepath, _img_path))]
    num_list = list(range(len(frame_list)))
    data_dict = xr.DataArray(
        frame_list,
        coords={'frame': num_list},
        dims=['frame', 'img_y', 'img_x'],
    )
    return data_dict

def split_val_dataset(seq, date, label):

    temp_label = None
    index_list = 2*[]
    index_len = []
    k = 0
    j = 0
    for i in range(len(label)):
        if temp_label == None:
            temp_label = label[0]
            index_list += [0]
            j += 1

        elif temp_label == label[i]:
            index_list[k] += i
            j += 1 
        else:
            temp_label = label[i]
            k += 1
            index_list += [i]
            index_len.append(j)
            j = 0

    seq_train = []
    date_train = []
    label_train = []

    seq_val = []
    date_val = []
    label_val = []

    for j in range(len(index_list)):
        if index_len[i] > 3:

            num = index_len[i] - int(index_len[i]/10) + 1

            seq_train += [seq[i] for i in index_list[j][:num]]
            date_train += [date[i] for i in index_list[j][:num]]
            label_train += [label[i] for i in index_len[j][:num]]

            seq_val += [seq[i] for i in index_list[j][num:]]
            date_val += [date[i] for i in index_list[j][num:]]
            label_val += [label[i] for i in index_len[j][num:]]
        else:
            seq_train += [seq[i] for i in index_list[j]]
            date_train += [date[i] for i in index_list[j]]
            label_train += [label[i] for i in index_len[j]]

    return seq_train, date_train, label_train, seq_val, date_val, label_val

def load_data(config, dataset_path, resolution, dataset, pid_num, appendix, cache, pid_shuffle=False):
    seq_dir = list()
    date = list()
    view = list()
    label = list()
    condition = list()
    print("path", dataset_path)
    
    if config.data.dataset == "OURS": # for ReSGait.
        for _label in sorted(list(os.listdir(dataset_path))):
            label_path = osp.join(dataset_path, _label)
            for seqs in sorted(list(os.listdir(label_path))):
                if appendix is not None:
                    seqs_path = osp.join(label_path, seqs,appendix)
                else:
                    seqs_path = osp.join(label_path, seqs)

                seq_dir.append([seqs_path])
                int_label = int(_label)
                label.append(int_label)
                date.append(seqs)

    elif config.data.dataset == "pose_OUMVLP":
        data_list_temp = sorted(list(os.listdir(dataset_path)))[100:]
        for _label in data_list_temp:
            label_path = osp.join(dataset_path, _label)
            for seqs in sorted(list(os.listdir(label_path))):

                if appendix is not None:
                    seqs_path = osp.join(label_path, seqs,appendix)
                else:
                    seqs_path = osp.join(label_path, seqs)

                seq_dir.append([seqs_path])
                int_label = int(_label)
                label.append(int_label)
                date.append(seqs)
    else:
        # skip the first 100, because there are some data missing.
        Label_range = sorted(list(os.listdir(dataset_path)))[100:]   
        print(Label_range[:10])
        for _label in Label_range:
            label_path = osp.join(dataset_path, _label)  # 005
            for cond_i in sorted(list(os.listdir(label_path))): # 01
                cond_path = osp.join(label_path, cond_i)

                # this code below is only for OUMVLP
                # skip the empty folder
                if len(cond_i) == 1:
                    # print("skip path: ", cond_path)  -
                    continue

                for view_i in sorted(list(os.listdir(cond_path))):
                    seqs_path = osp.join(cond_path, view_i)

                    seq_dir.append([seqs_path])
                    int_label = int(_label)
                    label.append(int_label)
                    view.append(view_i)
                    condition.append(cond_i)
                # for mat data, we need to process the tial of the file

    # seq_dir = list()
    # date = list()
    # label = list()

    pid_list = sorted(list(set(label)))
    pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
    train_list = pid_list[0]
    test_list = pid_list[1]

    if pid_shuffle:
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

    if config.data.dataset == "OURS" or config.data.dataset == "pose_OUMVLP":
        print("train in ", config.data.dataset)
        if config.train.validation:

            seq = [seq_dir[i] for i, l in enumerate(label) if l in train_list]
            date = [date[i] for i, l in enumerate(label) if l in train_list]
            label = [label[i] for i, l in enumerate(label) if l in train_list]

            seq_train, date_train, label_train, seq_val, date_val, label_val = split_val_dataset(seq, date, label)
            train_source = get_dataset(seq_train, date_train, label_train, config)
            val_source = get_dataset(seq_val, date_val, label_val, config)

        else:
            train_source = get_dataset(
                [seq_dir[i] for i, l in enumerate(label) if l in train_list],
                [date[i] for i, l in enumerate(label) if l in train_list],
                [label[i] for i, l in enumerate(label) if l in train_list],
                config)

            val_source = None
        if config.data.dataset == "pose_OUMVLP":
            test_source = None
        else:
            test_source = get_dataset(
                [seq_dir[i] for i, l in enumerate(label) if l in test_list],
                [date[i] for i, l in enumerate(label) if l in test_list],
                [label[i] for i, l in enumerate(label) if l in test_list],
                config)
    else:
        print("the dataset is OUMVLP!!")
        train_source = get_dataset(
            [seq_dir[i] for i, l in enumerate(label) if l in train_list],
            [condition[i] for i, l in enumerate(label) if l in train_list],
            [label[i] for i, l in enumerate(label) if l in train_list],
            config,
            view=[view[i] for i, l in enumerate(label) if l in train_list])

        val_source = None

        test_source = get_dataset(
            [seq_dir[i] for i, l in enumerate(label) if l in test_list],
            [condition[i] for i, l in enumerate(label) if l in test_list],
            [label[i] for i, l in enumerate(label) if l in test_list],
            config,
            view=[view[i] for i, l in enumerate(label) if l in test_list])

    return train_source, test_source, val_source, train_list



def load_data_for_covariate(config, dataset_path, resolution, dataset, pid_num, appendix, cache, pid_shuffle=False):
    seq_dir = list()
    date = list()
    view = list()
    label = list()
    condition = list()
    print("path", dataset_path)
    # load all data from file path

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for seqs in sorted(list(os.listdir(label_path))):

            if appendix is not None:
                # print("appendix", appendix)
                seqs_path = osp.join(label_path, seqs,appendix)
            else:
                seqs_path = osp.join(label_path, seqs)

            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            date.append(seqs)

    label_list = int_list(label)
    pid_list = sorted(list(set(label)))
    pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
    train_list = pid_list[0]
    test_list = pid_list[1]

    if config.data.dataset == "OURS":
        print("train in OURS")
        if config.train.validation:

            seq = [seq_dir[i] for i, l in enumerate(label) if l in train_list]
            date = [date[i] for i, l in enumerate(label) if l in train_list]
            label = [label[i] for i, l in enumerate(label) if l in train_list]

            seq_train, date_train, label_train, seq_val, date_val, label_val = split_val_dataset(seq, date, label)
            train_source = get_dataset(seq_train, date_train, label_train, config)
            val_source = get_dataset(seq_val, date_val, label_val, config)

        else:
            train_source = get_dataset(
                [seq_dir[i] for i, l in enumerate(label) if l in train_list],
                [date[i] for i, l in enumerate(label) if l in train_list],
                [label[i] for i, l in enumerate(label) if l in train_list],
                config)

            val_source = None

        test_source = get_dataset(
            [seq_dir[i] for i, l in enumerate(label) if l in test_list],
            [date[i] for i, l in enumerate(label) if l in test_list],
            [label[i] for i, l in enumerate(label) if l in test_list],
            config)
    else:
        print("the dataset is OUMVLP!!")
        print()
        train_source = get_dataset(
            [seq_dir[i] for i, l in enumerate(label) if l in train_list],
            [condition[i] for i, l in enumerate(label) if l in train_list],
            [label[i] for i, l in enumerate(label) if l in train_list],
            config,
            view=[view[i] for i, l in enumerate(label) if l in train_list])

        val_source = None

        test_source = get_dataset(
            [seq_dir[i] for i, l in enumerate(label) if l in test_list],
            [condition[i] for i, l in enumerate(label) if l in test_list],
            [label[i] for i, l in enumerate(label) if l in test_list],
            config,
            view=[view[i] for i, l in enumerate(label) if l in test_list])

    return train_source, test_source, val_source, train_list

def save_data_to_pickle(data, data_path):
    print("begin save data.")
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)
    # pickle_data.close()
    print("saving complete!!")

def load_all_data_frome_pickle(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    print("load data complete!!")
    return data

def initialize_data(config, train=False, test=False):
    print("Initializing data source...")
    train_source, test_source, val_source, train_list = load_data(config, config.data.dir, config.data.resolution, config.data.name, config.data.pid_num, config.data.appendix,cache=(train or test))

    if train:
        if config.data.dataset == "OURS":
            if config.if_train:
                data_path = os.path.join(config.data.cache_path, config.data.name+ "_T" +"_"+"train.npy")
            else:
                data_path = os.path.join(config.data.cache_path, config.data.name + "_N" + "_" + "train.npy")

        else:
            if config.if_train:
                data_path = os.path.join(config.data.cache_path, config.data.name + "_" + config.data.dataset + "_T" +"_"+"train.npy")
            else:
                data_path = os.path.join(config.data.cache_path, config.data.name + "_" + config.data.dataset + "_N" + "_" + "train.npy")
            print("OUMVL path is ", data_path)

        if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
            print("Loading training data from pickle")
            train_source = load_all_data_frome_pickle(data_path)

        else:
            print("Loading training data...")
            train_source.load_all_data()
            save_data_to_pickle(train_source, data_path)

    if test:
        if config.data.dataset == "OURS":
            data_path = os.path.join(config.data.cache_path, config.data.name+"_"+"test.npy")
        else:
            data_path = os.path.join(config.data.cache_path, config.data.name+"_" + config.data.dataset + "_" + "test.npy")
        if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
            print("Loading training data from pickle")
            test_source = load_all_data_frome_pickle(data_path)

        else:
            print("Loading training data...")
            test_source.load_all_data()
            save_data_to_pickle(test_source, data_path)

    print("Data initialization complete.")
    return train_source, test_source, val_source,train_list


def get_initial(config, train=True, test=False):
    # return data_loader
    print("Initialzing...")
    WORK_PATH = config.WORK_PATH
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

    if config.time_span:
        if test:
            print("test ...")
            test_source, test_gallery, test_list = load_data_from_time_span(config, config.data.dir, config.data.name, config.data.appendix, cache=(train or test), pid_shuffle=config.data.pid_shuffle, test=True)
            return test_source, test_gallery, test_list
        else:
            print(" train without val")
            train_source, test_source, train_list = load_data_from_time_span(config, config.data.dir, config.data.name, config.data.appendix, cache=(train or test), pid_shuffle=config.data.pid_shuffle)
        val_source = None
    
    else:
        print( "train with val")
        train_source, test_source, val_source, train_list = initialize_data(config, train, test)

    return train_source, test_source, val_source, train_list

def get_gallery_random(test_list, seq_dir, date, label, config):

    seq_dir_gallery = list()
    date_gallery = list()
    label_gallery = list()

    seq_dir_probe = list()
    date_probe = list()
    label_probe = list()

    seq_dir_temp = list()
    date_temp = list()
    label_temp = list()

    label_temp_ = None
    for i, l in enumerate(label):
        if l in test_list:
            if label_temp_ is None:

                seq_dir_temp.append(seq_dir[i])
                date_temp.append(date[i])
                label_temp.append(label[i])

                label_temp_ = label[i]

            elif label_temp_ is label[i]:

                seq_dir_temp.append(seq_dir[i])
                date_temp.append(date[i])
                label_temp.append(label[i])

            else:
                len_temp = len(label_temp)
                random_index = random.randint(0, len_temp-1)

                for k in range(len_temp):
                   if k is not random_index:

                        seq_dir_probe.append(seq_dir_temp[k])
                        date_probe.append(date_temp[k])
                        label_probe.append(label_temp[k])

                   else:

                       seq_dir_gallery.append(seq_dir_temp[k])
                       date_gallery.append(date_temp[k])
                       label_gallery.append(label_temp[k])

                seq_dir_temp = [seq_dir[i]]
                date_temp = [date[i]]
                label_temp = [label[i]]

                label_temp_ = label[i]

    # adding the last one
    len_temp = len(label_temp)
    print(len_temp)
    random_index = random.randint(0, len_temp - 1)

    for k in range(len_temp):
        if k is not random_index:

            seq_dir_probe.append(seq_dir_temp[k])
            date_probe.append(date_temp[k])
            label_probe.append(label_temp[k])

        else:

            seq_dir_gallery.append(seq_dir_temp[k])
            date_gallery.append(date_temp[k])
            label_gallery.append(label_temp[k])

    test_dataset = get_dataset(seq_dir_probe, date_probe, label_probe, config)
    test_gallery = [seq_dir_gallery, date_gallery, label_gallery]
    return test_dataset, test_gallery


def get_gallery_fix(test_list, seq_dir, date, label, config):

    seq_dir_gallery = list()
    date_gallery = list()
    label_gallery = list()

    seq_dir_probe = list()
    date_probe = list()
    label_probe = list()

    label_temp = None

    for i, l in enumerate(label):
        if l in test_list:
            if label_temp is None or label_temp is not label[i]:
                seq_dir_gallery.append(seq_dir[i])
                date_gallery.append(date[i])
                label_gallery.append(label[i])

                label_temp = label[i]
            else:
                seq_dir_probe.append(seq_dir[i])
                date_probe.append(date[i])
                label_probe.append(label[i])

    test_dataset = get_dataset(seq_dir_probe, date_probe, label_probe, config)
    test_gallery = [seq_dir_gallery, date_gallery, label_gallery]
    return test_dataset, test_gallery

def int_list(temp_list):
    temp = []
    for i in range(len(temp_list)):
        temp.append(int(temp_list[i]))
    return temp
                   
def load_data_test(config, dataset_path, dataset, pid_num, appendix, pid_shuffle=False):

    seq_dir = list()
    date = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for seqs in sorted(list(os.listdir(label_path))):
            if appendix is not None:
                seqs_path = osp.join(label_path, seqs, appendix)
            else:
                seqs_path = osp.join(label_path, seqs)
            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            date.append(seqs)

    # for mat data, we need to process the tial of the file
    if dataset == "pose":
        date = [date[i][:-4] for i in range(len(date))]

    pid_list = sorted(list(set(label)))
    # pid_list_test = pid_list[pid_num:]
    pid_list_test = test_id

    if config.time_span:
        pid_list_test = int_list(big_day)
    else:
        pid_list_test = int_list(pid_list_test)
    # next we need to creat a dataset class
    # get gallery sample

    get_gallery = globals().get("get_gallery_"+config.test.gallery_model)
    test_source, test_gallery = get_gallery(pid_list_test, seq_dir, date, label, config)
    return test_source, test_gallery


def get_gallery_data(test_dataset, test_gallery):

    seq_dir_gallery, date_gallery, label_gallery = test_gallery[0],test_gallery[1], test_gallery[2]

    data_gallery = list()

    for i in range(len(label_gallery)):
        data_gallery.append(test_dataset.loader(seq_dir_gallery[i]))

    test_gallery = [data_gallery, date_gallery, label_gallery]

    return test_gallery


def get_initial_test(config, train= False, test= True ):
    print("Initialzing test dataset...")
    test_source, test_gallery = load_data_test(config, config.data.dir, config.data.name, config.data.pid_num, config.data.appendix)
    print("Loading testing data...")

    test_source.load_all_data()

    # according to seq_gallery, load gallery data
    test_gallery = get_gallery_data(test_source, test_gallery)
    print("len probe set = ", len(test_source), ", len gallery set = ", len(test_gallery[2]))
    return test_source, test_gallery


def find_list_from_path(dataset_path, appendix):
    seq_dir = list()
    date = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for seqs in sorted(list(os.listdir(label_path))):
            if appendix is not None:
                seqs_path = osp.join(label_path, seqs, appendix)
            else:
                seqs_path = osp.join(label_path, seqs)
            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            date.append(seqs)

    return seq_dir, date, label

def get_initial_test_for_covariate_pose(config, appendix = None, task = "normal", probe_only = True):
    root_path = "PATH_FOR_TEST_DATA" 
    gallery_path = "PATH_FOR_GALLERY_DATA"

    if os.path.exists(root_path) and os.path.exists(gallery_path):
        traceback.print_exc("Error Path in func of get_initial_test_for_covariate_pose in \"./utils/initialization.py\"!!")

    probe_path = os.path.join(root_path, task)
    print("probe path =", probe_path)
    seq_dir_probe, date_probe, label_probe = find_list_from_path(probe_path, appendix)
    seq_dir_gallery, date_gallery, label_gallery = find_list_from_path(gallery_path, appendix)

    label_probe = int_list(label_probe)
    label_gallery = int_list(label_gallery)

    dataset = config.data.name
    # for mat data, we need to process the tial of the file
    if dataset == "pose":
        date_gallery= [date_gallery[i][:-4] for i in range(len(date_gallery))]
        date_probe= [date_probe[i][:-4] for i in range(len(date_probe))]

    pid_list_gellery = sorted(list(set(label_gallery)))
    pid_list_probe = sorted(list(set(label_probe)))

    # get_gallery = globals().get("get_gallery_"+config.test.gallery_model)
    # test_source, test_gallery = get_gallery(pid_list_test, seq_dir, date, label, config)

    test_source = get_dataset(seq_dir_probe, date_probe, label_probe, config)
    test_source.load_all_data()

    if probe_only:
        test_gallery = None
    else:
        data_gallery = list()

        for i in range(len(seq_dir_gallery)):
            data_gallery.append(load_data_from_path_pose(seq_dir_gallery[i][0]))

        test_gallery = [data_gallery, date_gallery, label_gallery]
    return test_source, test_gallery


def get_initial_test_for_covariate(config, appendix, task = "normal", probe_only = True):
    root_path = "PATH_FOR_TEST_DATA" 
    gallery_path = "PATH_FOR_GALLERY_DATA"

    if os.path.exists(root_path) and os.path.exists(gallery_path):
        traceback.print_exc("Error Path in func of get_initial_test_for_covariate in \"./utils/initialization.py\"!!")

    probe_path = os.path.join(root_path, task)
    print("probe path =", probe_path)
    seq_dir_probe, date_probe, label_probe = find_list_from_path(probe_path, appendix)
    seq_dir_gallery, date_gallery, label_gallery = find_list_from_path(gallery_path, appendix)

    label_probe = int_list(label_probe)
    label_gallery = int_list(label_gallery)

    dataset = config.data.name
    # for mat data, we need to process the tial of the file
    if dataset == "pose":
        date_gallery= [date_gallery[i][:-4] for i in range(len(date_gallery))]
        date_probe= [date_probe[i][:-4] for i in range(len(date_probe))]

    pid_list_gellery = sorted(list(set(label_gallery)))
    pid_list_probe = sorted(list(set(label_probe)))

    # get_gallery = globals().get("get_gallery_"+config.test.gallery_model)
    # test_source, test_gallery = get_gallery(pid_list_test, seq_dir, date, label, config)

    test_source = get_dataset(seq_dir_probe, date_probe, label_probe, config)
    test_source.load_all_data()

    if probe_only:
        test_gallery = None
    else:
        data_gallery = list()

        for i in range(len(seq_dir_gallery)):
            data_gallery.append(load_data_from_path(seq_dir_gallery[i]))

        test_gallery = [data_gallery, date_gallery, label_gallery]
    return test_source, test_gallery


def get_gallery_data_loader(config, train= False, test= True):
    test_source, test_gallery = load_data_test(config, config.data.dir, config.data.name, config.data.pid_num, config.data.appendix)
    print("Loading finte tuning data...")
    test_gallery_dataset = get_dataset(*test_gallery, config)  # use gallery dataset instant to present gallery
    test_gallery_dataset.load_all_data()

    return test_gallery_dataset


def get_initial_test_save(config, train= False, test= True ):
    print("Initialzing test dataset...")
    test_source, test_gallery = load_data_test(config, config.data.dir, config.data.name, config.data.pid_num, config.data.appendix)
    print("Loading testing data...")

    data_path = os.path.join(config.data.cache_path, config.data.name + "_" + "test.npy")

    if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
        print("Loading training data from pickle")
        test_source = load_all_data_frome_pickle(data_path)

    else:
        print("Loading test data from raw...")
        test_source.load_all_data()
        save_data_to_pickle(test_source, data_path)

    # test_source.load_all_data()
    # according to seq_gallery, load gallery data
    test_gallery = get_gallery_data(test_source, test_gallery)

    return test_source, test_gallery


def load_data_from_time_span(config, dataset_path, dataset, appendix, cache, pid_shuffle=False, test = False):
    seq_dir = list()
    date = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)        # 005
        for seqs in sorted(list(os.listdir(label_path))):
            if appendix is not None:
                seqs_path = osp.join(label_path, seqs,appendix)
            else:
                seqs_path = osp.join(label_path, seqs)

            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            date.append(seqs) 
    
    # for mat data, we need to process the tial of the file
    if dataset == "pose":
        date = [date[i][:-4] for i in range(len(date))]

    # seq_dir = list()
    # date = list()
    # label = list()

    train_list, test_list = map(int_list, [small_day, big_day])
    print(train_list, test_list)

    if pid_shuffle:
        np.random.shuffle(train_list)
        # np.random.shuffle(test_list)

    train_source = get_dataset(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [date[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        config)

    val_source = None
    test_source = None

    data_path = os.path.join(config.data.cache_path, config.data.name+"_"+"train_time_span.npy")

    if test:
        get_gallery = globals().get("get_gallery_"+config.test.gallery_model)

        test_source, test_gallery = get_gallery(test_list, seq_dir, date, label, config)
        test_source.load_all_data()
        test_gallery = get_gallery_data(test_source, test_gallery)
        return test_source, test_gallery, test_list

    else:
        if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
            print("Loading training data from pickle")
            train_source = load_all_data_frome_pickle(data_path)

        else:
            print("Loading training data...")
            train_source.load_all_data()
            save_data_to_pickle(train_source, data_path)

    print("Data initialization complete.")
    return train_source, test_source, train_list
