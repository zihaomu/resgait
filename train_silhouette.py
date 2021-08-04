from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm
import sys

# sys.path.append('configs')

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import get_model
from losses import get_loss, get_center_loss
from optimizers import get_optimizer, get_center_optimizer
from schedulers import get_scheduler
from sampler import get_sampler

import utils
from utils.checkpoint import get_initial_checkpoint, load_checkpoint, save_checkpoint
import utils.metrics
from utils import get_initial, get_collate_fn


from sklearn.preprocessing import LabelEncoder
# change training parameters from py dictionary to 

class Train(object): 

    def __init__(self, config): 

        self.config = config
        self.model = None
        self.optimizer = None
        self.optimizer_center = None  # reserved for center loss
        self.scheduler = None
        self.writer = None
        self.label = None
        self.label_encoder = None
        self.sampler = None
        self.loss_function = None
        self.loss_center = None       # reserved for center loss
        self.writer = None
        self.data_loader = None
        self.data_loader_val = None
        self.dataset = None
        self.more_label = None
        self.collate_fn = None
        self.weight = self.config.train.weight
        self.num_epochs = self.config.train.num_epochs
        self.num_workers = self.config.data.num_workers
        self.sample_type = 'random'
        self.last_epoch = 0
        self.step = -1
        self.iteration = 0
        if self.writer is not None:
            self.writer = SummaryWriter(self.config.writer)

        self.more_label = self.load_new_label()

    def load_new_label(self):
        data = pd.read_csv("./data/label.csv")
        data = data.drop(columns=['ID'])
        return data

    def initialization(self):
        self.dataset, test_dataset, val_dataset, self.label = get_initial(self.config, train = True)  # return dataset instance
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label)
        torch.cuda.empty_cache()
        self.model = get_model(self.config)
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        checkpoint = get_initial_checkpoint(self.config)

        if torch.cuda.device_count() >1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        if checkpoint is not None:
            self.last_epoch, self.step = load_checkpoint(self.model, self.optimizer, self.optimizer_center, checkpoint)
        print("from checkpoint {} last epoch: {}".format(checkpoint, self.last_epoch))

        self.sampler = get_sampler(self.dataset, self.config) 
        self.loss_function = get_loss(self.config)


        if self.config.loss.name is 'softmax_center':
            self.loss_center = get_center_loss(class_num = 86, feature_num = 512)
            self.optimizer_center = get_center_optimizer(self.loss_center.parameters(), self.config.optimizer.params.lr)

        self.collate_fn = get_collate_fn(self.config, self.config.data.frame_num, self.sample_type) #

        if self.sampler is not None:
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_sampler=self.sampler,
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers)

        else:
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.config.train.batch_size.batch1,
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers,
                drop_last= self.config.data.drop_last,
                shuffle= self.config.data.pid_shuffle,
            )


    def validation(self, epoch):
        self.model.eval()

        with torch.no_grad():

            epoch_val = 10
            acc_sample = 0
            count_all = 0
            all_loss = 0

            for epoch_i in range(epoch_val):
                for seq, date, label, _ in self.data_loader_val:
                    count_all += len(label)

                    seq = torch.Tensor(seq).float().cuda()
                    fc, out = self.model(seq)

                    label = self.label_encoder.transform(label)
                    label = torch.Tensor(label).long().cuda()

                    loss = self.loss_function(out, label)
                    pred = torch.max(out, 1)[1]
                    acc = (pred == label).sum()

                    acc_sample += acc.item()
                    all_loss += loss
            acc_ii = acc_sample/count_all
            if self.writer is not None:
                self.writer.add_scalar("data/val_loss", all_loss, epoch)
                self.writer.add_scalar("data/val_acc", acc_ii, epoch)
            
            print("validated result, in epoch :{}, acc = {}, loss={}".format(epoch, acc_ii, all_loss))
        
        self.model.train()

    def find_label(self, date, label):
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

        for i in range(len(date)):

            value = self.more_label.loc[(self.more_label['id'] == label[i]) & (self.more_label['date'] == date[i]) ].values[0]
            cloth.append(int(value[0]))
            activity.append(int(value[1]))
            gender.append(int(value[2]))
            carry.append(int(value[3]))
            path.append(int(value[4]))

        cloth = np.asarray(cloth)
        activity = np.asarray(activity)
        gender = np.asarray(gender)
        carry = np.asarray(carry)
        path = np.asarray(path)

        # print(cloth)
        return cloth,activity,gender,carry,path

    def train_sigle_iteration(self, seq, date, label, _):

        self.optimizer.zero_grad()
        if self.optimizer_center is not None:
            self.optimizer_center.zero_grad()

        # generate new label
        cloth, activity, gender, carry, path = self.find_label(date, label)

        seq = torch.Tensor(seq).float().cuda()
        fc, out, out_cloth, out_activity, out_gender, out_carry, out_path = self.model(seq)

        label = self.label_encoder.transform(label)
        label = torch.Tensor(label).long().cuda()
        cloth = torch.Tensor(cloth).long().cuda()
        activity = torch.Tensor(activity).long().cuda()
        gender = torch.Tensor(gender).long().cuda()
        carry = torch.Tensor(carry).long().cuda()
        path = torch.Tensor(path).long().cuda()

        loss = self.loss_function(out, label) # loss of identification.
        pred = torch.max(out, 1)[1]
        acc = (pred == label).sum()

        loss_cloth = self.loss_function(out_cloth, cloth)
        loss_activity = self.loss_function(out_activity, activity)
        loss_gender = self.loss_function(out_gender, gender)
        # loss_carry = self.loss_function(out_carry, carry)
        # loss_path = self.loss_function(out_path, path)

        # NOTE:The following code is for Experiments of gait covariate.
        # You can customize the following code for you case.
        loss = loss + self.weight*loss_cloth +self.weight*loss_activity + self.weight*loss_gender #+ loss_carry + loss_path
        loss_temp = loss.item()

        loss.backward()

        self.optimizer.step()  # update parameters
        if self.loss_center is not None:
            self.optimizer_center.step()
        return acc.item(), loss_temp

    def train_weigh(self):
        acc_sample = 0
        count_all = 0
        all_loss = 0

        total_num = len(self.dataset)
        batch_size = self.config.train.batch_size.batch1 * self.config.train.batch_size.batch2

        step_num = math.ceil(total_num/batch_size)

        epoch = self.last_epoch
        iteration = epoch*step_num
        # print("step number is ", step_num)
        for seq, date, label, _ in self.data_loader:
            iteration += 1

            count_all += len(label)

            acc_i, loss = self.train_sigle_iteration(seq, date, label, _)
            all_loss += loss
            acc_sample += acc_i
            # print("iteration is {}, epoch is {}".format(iteration, epoch))
            if iteration % step_num == step_num-1 :
                epoch += 1
                if self.writer is not None:
                    self.writer.add_scalar("train_loss", all_loss, epoch)
                acc_epoch = acc_sample * 1.0 / count_all
                print("training in epoch :{}, the acc is {}% ,\n the loss is {}".format(epoch, acc_epoch * 100, all_loss))
                acc_sample = 0
                count_all = 0
                all_loss = 0

                # if epoch % 10 == 9:
                #     self.validation(epoch)

            if epoch > self.config.train.num_epochs:
                break

            if epoch % 200 == 199:
                save_checkpoint(self.config, self.model, self.optimizer, self.optimizer_center, epoch, self.step)


    def train_single_epoch(self, epoch ):
        acc_sample = 0
        count_all = 0
        all_loss = 0

        for seq, date, label, _ in self.data_loader:
            count_all += len(label)
            acc_i, loss = self.train_sigle_iteration(seq, date, label, _)
            all_loss += loss
            acc_sample += acc_i
        if self.writer is not None:
            self.writer.add_scalar("train_loss", all_loss, epoch)
        acc_epoch = acc_sample * 1.0 / count_all

        print("training in epoch :{}, the acc is {}% ,\n the loss is {}".format(epoch, acc_epoch * 100, all_loss))
        print("learning rate: ",self.optimizer.param_groups[0]['lr'])


    def run(self):
        # checkpoint
        self.scheduler = get_scheduler(self.config, self.optimizer)
        self.model.train()
        postfix_dic = {
            'lr': 0.0,
            'acc' : 0.0,
            'loss' : 0.0,
        }

        if self.config.data.sampler== "weight":
            self.train_weigh()
        else:
            for epoch in range(self.last_epoch, self.num_epochs):

                self.train_single_epoch(epoch)

                if epoch % 200 == 199:
                    save_checkpoint(self.config, self.model, self.optimizer, self.optimizer_center, epoch, self.step)

                # if epoch % 10 ==9:
                #     self.validation(epoch)

            # saving model every 200 epoch
                self.scheduler.step()
                if epoch > self.config.train.num_epochs:
                    break


def parse_args():
    parser = argparse.ArgumentParser(description='config file')

    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="./configs/YOUR_CONFIG.yml", type=str)
    
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
    
    if args.GPU_num is not None:
        config.CUDA_VISIBLE_DEVICES = args.GPU_num
        print("GPU is ", config.CUDA_VISIBLE_DEVICES)
    
    print(config.train.dir)
    trainer = Train(config)
    trainer.initialization()
    trainer.run()
    print("Training complete!")

if __name__ == '__main__':
    main()
