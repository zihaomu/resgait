import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np



class PoseNet(nn.Module):
    def __init__(self, frame_num = 28, num_classes = 86, x_dims = 36, y_dims = 0):  # our skeleton point is coco data set
        # feature dimension is determined by the function in model_factory
        super(PoseNet,self).__init__()
        self.x_dims = x_dims
        self.y_dims = y_dims
        temp = ((frame_num-4)/2 -2)/2
        self.feature_dims = int(7*128*(temp))
        print(frame_num,self.feature_dims)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #self.fc1 = nn.Linear(3584,512)  # frame = 25 -> ?????
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out1)   # 128x86
        return out1, out2

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class PoseNet_new(nn.Module):
    def __init__(self, frame_num = 28, num_classes = 86, x_dims = 36, y_dims = 0):  # our skeleton point is coco data set
        # feature dimension is determined by the function in model_factory
        super(PoseNet_new,self).__init__()
        self.x_dims = x_dims
        self.y_dims = y_dims
        temp = ((frame_num-4)/2 -2)/2
        self.feature_dims = int(7*128*(temp))
        print(frame_num,self.feature_dims)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #self.fc1 = nn.Linear(3584,512)  # frame = 25 -> ?????
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30

        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, 2)

    def get_feature_dim(self):
        return self.feature_dims

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1

        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out1)   # 128x86
        out3 = self.fc3(out1)
        return out1, out2, out3

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_pose_ft(nn.Module):
    def __init__(self, frame_num = 28, num_classes = 86, x_dims = 36, y_dims = 0):  # our skeleton point is coco data set
        # feature dimension is determined by the function in model_factory
        super(CNN_pose_ft,self).__init__()
        self.x_dims = x_dims
        self.y_dims = y_dims
        temp = ((frame_num-4)/2 -2)/2
        # self.feature_dims = int(7*128*(temp))
        self.feature_dims = 5376
        print(frame_num, self.feature_dims)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #self.fc1 = nn.Linear(3584,512)  # frame = 25 -> ?????
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30

        self.fc2 = nn.Linear(512, num_classes)
        self.gender = nn.Linear(512, 2)

    def get_feature_dim(self):
        return self.feature_dims

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out1)   # 128x86
        out3 = self.gender(out1)
        return out1, out2, out3

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class CNN_pose_new(nn.Module):
    def __init__(self, frame_num = 28, num_classes = 86, x_dims = 36, y_dims = 0):  # our skeleton point is coco data set
        # feature dimension is determined by the function in model_factory
        super(CNN_pose_new,self).__init__()
        self.x_dims = x_dims
        self.y_dims = y_dims
        temp = ((frame_num-4)/2 -2)/2
        self.feature_dims = 5376
        print(frame_num,self.feature_dims)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #self.fc1 = nn.Linear(3584,512)  # frame = 25 -> ?????
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30

        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, 2)

    def get_feature_dim(self):
        return self.feature_dims

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out1)   # 128x86
        out3 = self.fc3(out1)
        return out1, out2, out3

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PoseNet_ft(nn.Module):
    def __init__(self, frame_num = 28, num_classes = 86, x_dims = 36, y_dims = 0):  # our skeleton point is coco data set
        # feature dimension is determined by the function in model_factory
        super(PoseNet_ft, self).__init__()
        self.x_dims = x_dims
        self.y_dims = y_dims
        temp = ((frame_num-4)/2 -2)/2
        self.feature_dims = int(7*128*(temp))
        print(frame_num,self.feature_dims)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #self.fc1 = nn.Linear(3584,512)  # frame = 25 -> ?????
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30

        self.new_out = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, 2)

    def get_feature_dim(self):
        return self.feature_dims

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.new_out(out1)   # 128x86
        out3 = self.fc3(out1)
        return out1, out2, out3


    def num_flat_features(self, x):
        size = x.size()[1:]  # [B, C, H, W]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class PoseNet_no_normal(nn.Module):
    def __init__(self, frame_num= 28, num_classes = 86, x_dims = 36, y_dims = 0):
        super(PoseNet_no_normal,self).__init__()

        temp = ((frame_num-4)/2 -2)/2
        self.feature_dims = int(7*128*(temp))
        self.x_dims = x_dims
        self.y_dims = y_dims

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(self.feature_dims,512)   # frame = 30
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.block1(x)
        out1 = self.pool1(out)
        out = self.block2(out1) + out1
        out = self.block3(out)
        out1 = self.pool2(out)
        out = self.block4(out1) + out1
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out1)
        return out1, out2