from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
import pandas as pd
# sys.path.append('configs')

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from models import get_model
from losses import get_loss, get_center_loss
from optimizers import get_optimizer, get_center_optimizer
from schedulers import get_scheduler
from sampler import get_sampler

import utils
from utils.checkpoint import get_checkpoint, load_checkpoint, save_checkpoint
import utils.metrics
from utils import get_initial_test, get_collate_fn, get_gallery_data, evaluation, Evaluator, get_initial
from utils import L2_distance, Vector_module, update_gallery
# change training parameters from py dictionary to

class Test(object):

    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.optimizer_center = None  # reserved for center loss
        self.scheduler = None
        self.writer = None
        self.sampler = None
        self.loss_function = None
        self.center_model = None  # reserved for center loss
        # self.writer = self.config.writer
        self.writer = None
        self.data_loader = None
        self.dataset = None
        self.data_loader_test = None
        self.gallery = None
        self.collate_fn = None
        self.num_epochs = self.config.train.num_epochs
        self.num_workers = self.config.data.num_workers
        self.sample_type = 'all'
        self.last_epoch = 0
        self.step = -1
        self.more_label = self.load_new_label()
        self.iteration = 0

        if self.writer is not None:
            self.writer = SummaryWriter(self.config.writer)

    def initialization(self):
        WORK_PATH = self.config.WORK_PATH
        os.chdir(WORK_PATH)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        print("GPU is :",os.environ["CUDA_VISIBLE_DEVICES"] )
        # step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
        self.model = get_model(self.config)
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        checkpoint = get_checkpoint(self.config)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.last_epoch, self.step = load_checkpoint(self.model, self.optimizer, self.center_model, self.optimizer_center, checkpoint)
        print("from checkpoint {} last epoch: {}".format(checkpoint, self.last_epoch))

        # dangerous code
        # print("new model ",self.model.state_dict()['module.conv1.0.weight'])
        # exit()

        self.collate_fn = get_collate_fn(self.config, self.config.data.frame_num, self.sample_type)  #

    def load_new_label(self):
        data = pd.read_csv("./data/label.csv")
        data = data.drop(columns=['ID'])
        return data


    def find_new_label(self, date, label):
        # cloth:  normal, coat, skirt: 0,1,2
        # activity = walk, phone:0,1
        # gender = male, female : 0,1
        # carry = no, bag, small, big : 0,1,2,3
        # path = straight, curve :0,1
        # upper = short, long : 0,1

        cloth = []
        activity = []
        gender = []
        carry = []
        path = []
        # print('label ',self.more_label)

        for i in range(len(date)):

            value = self.more_label.loc[(self.more_label['id'] == label[i]) & (self.more_label['date'] == date[i]) ].values[0]
            # print(value)
            # print(label)
            # print(date)
            # print(self.more_label)
            cloth.append(value[0])
            activity.append(value[1])
            gender.append(value[2])
            carry.append(value[3])
            path.append(value[4])

        cloth = np.asarray(cloth)
        activity = np.asarray(activity)
        gender = np.asarray(gender)
        carry = np.asarray(carry)
        path = np.asarray(path)

        # print(cloth)
        return cloth,activity,gender,carry,path


    def pose_build_batch(self, mat_data):

        len_mat = mat_data.shape[1]
        if len_mat < self.config.data.frame_num:

            mat_data = np.pad(mat_data, ((0, 0), (0, self.config.data.frame_num - len_mat)), 'constant', constant_values = 0)
            data = torch.unsqueeze(torch.from_numpy(mat_data),0)

        else:
            j = 0
            data = torch.unsqueeze(torch.from_numpy(mat_data[:, j:j+self.config.data.frame_num]), 0)
            j += 10

            while j + self.config.data.frame_num < len_mat:
                data_temp = torch.unsqueeze(torch.from_numpy(mat_data[:, j:j+self.config.data.frame_num]), 0)
                j += 10
                data = torch.cat([data, data_temp], 0)

        data = data.float().cuda()
        fc, pre, _ = self.model(data)
        feature = torch.mean(fc, 0)

        return feature


    def extract_gallery_feature(self, data_gallery, len_gallery):

        features = list()

        if self.config.data.name == "pose":

            for i in range(len_gallery):
                mat_i = data_gallery[i]
                fc = self.pose_build_batch(mat_i)                   # return the mean feature
                feat = fc.view(1, -1).data.cpu().numpy()
                n = 1
                for ii in range(n):
                    feat[ii] = feat[ii] / np.linalg.norm(feat[ii])
                features.append(feat)
        else:

            for i in range(len_gallery):
                # print("len gallery = ", len(data_gallery), " ", len(data_gallery[i], " ", len(data_gallery[i][0])))
                if type(data_gallery) is np.ndarray:
                    seq = data_gallery[i]
                else:
                    seq = data_gallery[i].values
                seq = torch.from_numpy(np.asarray(seq))
                seq = torch.unsqueeze(seq, 0)
                # seq = [torch.Tensor(seq[i]).float().cuda() for i in range(len(seq))]

                fc, out, out_cloth, out_activity, out_gender, out_carry, out_path = self.model(seq)
                n, num_bin = fc.size()
                feat = fc.view(n, -1).data.cpu().numpy()

                # if needing normalization
                for ii in range(n):
                    feat[ii] = feat[ii] / np.linalg.norm(feat[ii])
                features.append(feat)

        return features

    # For drawing the gender ROC_EER.
    def save_gender(self, gender_list, label_list):
        np.save(os.path.join(self.config.train.dir, "gender.npy"), gender_list)
        np.save(os.path.join(self.config.train.dir, "label.npy"), label_list)
        print("save success!!" )
        
    def run(self):
        # checkpoint
        self.model = self.model.eval()
        self.dataset, test_gallery = get_initial_test(self.config, test=True)  # return dataset instance

        print("data set len is :",len(self.dataset))
        data_gallery, date_gallery, label_gallery = test_gallery[0], test_gallery[1], test_gallery[2]

        print("----------->",self.config.test.sampler)
        # define dataloader
        if self.config.test.sampler != 'seq':
            print(" sampler is video level")
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size= 1,
                sampler=SequentialSampler(self.dataset),
                collate_fn=self.collate_fn,
                num_workers=self.num_workers)

        else:
            print(" sampler is seq level")
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.config.train.batch_size.batch1,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                drop_last=False,
                shuffle=True,
            )

        # the following code is for expanding the gallery sequence by sliding window.
        # data_gallery, date_gallery, label_gallery = update_gallery(data_gallery, date_gallery, label_gallery,
        #                                                            frame_num=32, overlap_per=0.4)
        len_gallery = len(label_gallery)

        feature_gallery = self.extract_gallery_feature(data_gallery, len_gallery)

        probe_feature = list()
        probe_date = list()
        probe_label = list()

        gender_save = []
        label_save = []

        # because in pose_based experiment, the batch samples are random, we circle 50 times to balance that.
        epoch = 1
        iterater = range(epoch)
        if self.config.test.sampler == "seq":
            epoch = 100
            iterater = tqdm.tqdm(range(epoch))

        print("epoch is :",epoch)

        for kk in iterater:
            final_label = []
            final_date = []

            pre_cloth = []
            pre_activity = []
            pre_gender = []
            pre_carry = []
            pre_path = []

            label_cloth = []
            label_activity = []
            label_gender = []
            label_carry = []
            label_path = []

            for seq, date, label, _ in self.data_loader:
                cloth, activity, gender, carry, path = self.find_new_label(date, label)

                label_cloth.extend(cloth)
                label_activity.extend(activity)
                label_gender.extend(gender)
                label_carry.extend(carry)
                label_path.extend(path)

                seq = torch.from_numpy(seq).float().cuda()
                # print(seq.size())

                fc, out, out_gender = self.model(seq)
                # print("out shape =", out.size())
                pre_gender.extend(torch.max(out_gender, 1)[1].detach().cpu().numpy())

                gender_probability= F.softmax(out_gender, dim=1)

                gender_temp = gender_probability.detach().cpu().numpy()[:,0]
                temp_value = np.average(gender_temp)

                gender_save.append(temp_value)
                label_save.append(gender[0])

                n, num_bin = fc.size()
                feat = fc.view(n, -1).data.cpu().numpy()

                for ii in range(n):
                    feat[ii] = feat[ii] / np.linalg.norm(feat[ii])

                probe_feature.append(feat)
                probe_label += label
                probe_date += date

            # begin test

            def transform_to_numpy(temp):
                return np.asarray(temp)

            pre_gender, label_gender = map(transform_to_numpy,[pre_gender, label_gender])
            # pre_activity = np.asarray(pre_activity)
            # pre_gender = np.asarray
            # pre_carry = []
            # pre_path = []

            acc_gender = np.sum(pre_gender == label_gender) / float(len(label_cloth))

            def to_list(pre, label):
                result = []
                for i in range(len(pre)):
                    if pre[i] == label[i]:
                        result.append(0)
                    else:
                        result.append(1)

                return result

            cloth_list = to_list(pre_cloth, label_cloth)
            activity_list = to_list(pre_activity, label_activity)
            gender_list = to_list(pre_gender, label_gender)

            # self.write_txt(probe_label, probe_date,date_gallery, label_gallery,cloth_list, activity_list, gender_list)

            print('acc_cloth', acc_gender)

        gender_save = np.asarray(gender_save)#
        label_save = np.asarray(label_save)
        print(gender_save)
        print(label_save)
        self.save_gender(gender_save, label_save)

        test_gallery = feature_gallery, date_gallery, label_gallery
        test_probe = np.concatenate(probe_feature, 0), probe_date, probe_label

        evaluation = Evaluator(test_gallery, test_probe, self.config)

        return evaluation.run()

    def write_txt(self, label, date, gallery, gallery_label,cloth, activity,gender , temp = "soft"):
        file = None

        if self.config.test.result_save:
            txt_path = os.path.join(self.config.train.dir, temp+'.txt')
            print(txt_path)
            file = open(txt_path, "w")

        for i in range(len(label)):
            temp = gallery[gallery_label.index(label[i])]
            # print(temp)
            str_str = str(label[i]) + "," + str(date[i]) + "," + str(temp) + "," + str(cloth[i]) + "," + str(activity[i])+ ","+ str(gender[i])+"\n"
            file.write(str_str)

        if file is not None:
            file.close()

        print("write success!!")


    def inference(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='config file')

    # # transformer
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="./configs/pose.yml", type=str)

    parser.add_argument('--epoch', dest='epoch',
                        help='epoch',
                        default="749", type=str)

    parser.add_argument('--GPU_num', dest='GPU_num',
                        help='GPU number',
                        default="0", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file is None:
        raise Exception("no configuration file.")

    config = utils.config.load(args.config_file)
    config.train.dir = os.path.join(config.train.dir, os.path.basename(args.config_file)[:-4])

    if args.epoch is not None:
        config.test.epoch = int(args.epoch)
        print("Epoch ", config.test.epoch)

    if args.GPU_num is not None:
        config.CUDA_VISIBLE_DEVICES = args.GPU_num
        print("GPU is ", config.CUDA_VISIBLE_DEVICES)

    # config.if_train = False # True or False
    print(config.train.dir)
    trainer = Test(config)
    trainer.initialization()

    right_probe_top1 = 0
    right_probe_top5 = 0
    num_probe = 0

    if config.test.gallery_model == "random":
        # in random model, cycle the entire process 50 times
        for i in range(10):
            right_probe_top1_, right_probe_top5_, num_probe_ = trainer.run()
            right_probe_top1 += right_probe_top1_
            right_probe_top5 += right_probe_top5_
            num_probe += num_probe_

        print("\n \n the top1 accuracy is : {}%, \nthe rank 5 accuracy is {}%. ".format(right_probe_top1 * 100.0 / num_probe,
                                                                                  right_probe_top5 * 100.0 / num_probe))

    else:
       _, _, _  = trainer.run()
    print("Finishing Test!")


if __name__ == '__main__':
    main()
