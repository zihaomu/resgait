from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

# import other packages in this folder
from .pose import PoseDataSet
from .silhouette import SilhouetteDataSet, SilhouetteDataSet_OUMVLP


# For ReSGait Pose data.
def get_pose(seq_dir, date, label, config):
    dataset = PoseDataSet(seq_dir, date, label, config)
    return dataset

# For OUMVLP Pose data.
def get_pose_OUMVLP(seq_dir, date, label, config):
    dataset = PoseDataSet(seq_dir, date, label, config)
    return dataset

# For OUMVLP silhouette data.
def get_silhouette_OUMVLP(seq_dir, date, view, label, config):
    dataset = SilhouetteDataSet_OUMVLP(seq_dir, date, view, label, config)
    return dataset

# Fo ReSGait silhouette data.
def get_silhouette(seq_dir, date, label, config):
    dataset = SilhouetteDataSet(seq_dir, date, label, config)
    return dataset

# Fo CASIA-B silhouette data.
# def get_silhouette_CASIA_B(seq_dir, date, view, label, config):
#     dataset = SilhouetteDataSet_OUMVLP(seq_dir, date, view, label, config)
#     return dataset

def get_dataset(seq_dir, date, label, config, transform=None, view=None):
    f = globals().get('get_'+config.data.name) # begin creat a class
    if transform is None:
        pass
    if config.data.dataset == "OURS" or config.data.dataset == "pose_OUMVLP":
        return f(seq_dir, date, label, config)
    else:
        return f(seq_dir, date, view, label, config)
