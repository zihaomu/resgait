import torch
import torch.nn as nn
import numpy as np
import torchvision

# You can add more model at this file, and do not forget to add 
# the corresponding function interface at ./model_factory,
# than you can call new model at "./config/YOUR_CONFIG.yml".

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class SilhouetteDeep(nn.Module):

    def __init__(self, feature_dimension = 512 ,block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteDeep, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128),
                                    BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64),
                                    Bottleneck(256, 64),
                                    Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128),
                                    Bottleneck(512, 128),
                                    Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,feature_dimension)
        self.out = nn.Linear(feature_dimension, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)
        print("n : =", n)

        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            print("input size",input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)

            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp

        fc = torch.mean(out, 1)
        out = self.out(fc)

        return fc, out


class SilhouetteNormal(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,feature_dimension)
        self.out = nn.Linear(feature_dimension, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  #batch
        # print("------>", silho.size)
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out(fc)

        return fc, out


class SilhouetteDeep_new_label(nn.Module):

    def __init__(self, feature_dimension = 512 ,block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteDeep_new_label, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128),
                                    BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64),
                                    Bottleneck(256, 64),
                                    Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.BatchNorm2d(512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128),
                                    Bottleneck(512, 128),
                                    Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,feature_dimension)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  # batch

        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)

            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp

        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1

        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path


class SilhouetteNormal_new_label(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal_new_label, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.BatchNorm2d(512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024,feature_dimension)
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        # print("in model sub data", silho)
        # print("in model sub",silho.size())
        n = silho.size(1)  #batch
        # print("------>", silho.size())
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            # print("input size", input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path


class SilhouetteNormal_new_label_ft(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal_new_label_ft, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.BatchNorm2d(512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024,feature_dimension)
        self.out_label_new = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        # print("in model",silho.size())
        n = silho.size(1)  #batch
        # print("------>", silho.size())
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            # print("input size", input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out_label_new(fc)
        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path


class SilhouetteNormal_new_label_div(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86, div_channel = 1):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal_new_label_div, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,int(64/div_channel)),
                                   nn.BatchNorm2d(int(64/div_channel)),
                                   conv3x3(int(64/div_channel), int(64/div_channel)),
                                   nn.BatchNorm2d(int(64/div_channel)),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(int(64/div_channel), int(64/div_channel))
        self.conv2 = nn.Sequential(conv3x3(int(64/div_channel),int(128/div_channel)),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(int(128/div_channel), int(128/div_channel)))
        self.conv3 = nn.Sequential(conv3x3(int(128/div_channel),int(256/div_channel)),
                                   nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(Bottleneck(int(256/div_channel), int(64/div_channel)))
        self.conv4 = nn.Sequential(conv3x3(int(256/div_channel),int(512/div_channel)),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(int(512/div_channel), int(128/div_channel)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(int(512/div_channel),feature_dimension)
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  #batch
        # print("------>", silho.size())
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            # print("input size", input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # print("print ",x.size())
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path



class SilhouetteDeep_new_label_div(nn.Module):

    def __init__(self, feature_dimension = 512 ,block = Bottleneck, num_classes=86, div_channel = 1):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteDeep_new_label_div, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,int(64/div_channel)),
                                   nn.BatchNorm2d(int(64/div_channel)),
                                   conv3x3(int(64/div_channel), int(64/div_channel)),
                                   nn.BatchNorm2d(int(64/div_channel)),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(int(64/div_channel), int(64/div_channel))
        self.conv2 = nn.Sequential(conv3x3(int(64/div_channel),int(128/div_channel)),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(int(128/div_channel), int(128/div_channel)),
                                    BasicBlock(int(128/div_channel), int(128/div_channel)))
        self.conv3 = nn.Sequential(conv3x3(int(128/div_channel),int(256/div_channel)),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(int(256/div_channel), int(64/div_channel)),
                                    Bottleneck(int(256/div_channel), int(64/div_channel)),
                                    Bottleneck(int(256/div_channel), int(64/div_channel)))
        self.conv4 = nn.Sequential(conv3x3(int(256/div_channel),int(512/div_channel)),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(int(512/div_channel), int(128/div_channel)),
                                    Bottleneck(int(512/div_channel), int(128/div_channel)),
                                    Bottleneck(int(512/div_channel), int(128/div_channel)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024/div_channel),feature_dimension)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  # batch

        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)

            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp

        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1

        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path

class SilhouetteNormal_new_label8(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal_new_label8, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,8),
                                   nn.BatchNorm2d(8),
                                   conv3x3(8, 8),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(8, 8)
        self.conv2 = nn.Sequential(conv3x3(8,16),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(16, 16))
        self.conv3 = nn.Sequential(conv3x3(16,32),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(32, 8))
        self.conv4 = nn.Sequential(conv3x3(32,64),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(64, 16))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128,feature_dimension)
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  #batch
        # print("------>", silho.size())
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            # print("input size", input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            # print("x1 :", x.size())
            x = self.layer4(x)

            # print("x : ", x.size())
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path


class SilhouetteDeep_new_label8(nn.Module):

    def __init__(self, feature_dimension = 512 ,block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteDeep_new_label8, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,8),
                                   nn.BatchNorm2d(8),
                                   conv3x3(8, 8),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(8, 8)
        self.conv2 = nn.Sequential(conv3x3(8,16),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(16, 16),
                                    BasicBlock(16, 16))
        self.conv3 = nn.Sequential(conv3x3(16,32),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(32, 8),
                                     Bottleneck(32, 8),
                                     Bottleneck(32, 8),
                                     Bottleneck(32, 8),
                                     Bottleneck(32, 8))
        self.conv4 = nn.Sequential(conv3x3(32,64),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(64, 16),
                                    Bottleneck(64, 16),
                                    Bottleneck(64, 16))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128,feature_dimension)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1
        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  # batch

        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)
            # print("X : ",x.size())
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)

            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp

        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1

        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path

class Finetuning(nn.Module):

    def __init__(self,feature_dimension = 512, num_classes=86, model=None):
        super(Finetuning, self).__init__()
        print(model)
        self.model = nn.Sequential(*list(model.modules())[:-8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = list(model.modules())[53]

        self.out_label = nn.Linear(feature_dimension, num_classes)
        self.out_cloth = nn.Linear(feature_dimension, 3)
        self.out_activity = nn.Linear(feature_dimension, 2)
        self.out_carry = nn.Linear(feature_dimension, 4)
        self.out_gender = nn.Linear(feature_dimension, 2)
        self.out_path = nn.Linear(feature_dimension, 2)

    def forward(self, silho):
        print("input :", silho.size())
        out = self.model(silho)
        print("x",out.size())

        fc = torch.mean(out, 1)
        out = self.out_label(fc)
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1

        out_cloth = self.out_cloth(fc)
        out_activity = self.out_activity(fc)
        out_gender = self.out_gender(fc)
        out_carry = self.out_carry(fc)
        out_path = self.out_path(fc)

        return fc, out, out_cloth, out_activity, out_gender, out_carry, out_path



if __name__=="__main__":
    model = SilhouetteNormal_new_label()
    new_model = Finetuning(model=model)
    print(new_model)