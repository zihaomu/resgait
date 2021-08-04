import os
import torch
import torch.nn.functional as F
import numpy as np

# write result to txt

def cuda_dist(x, y):  # probe_seq_x, gallery_seq_y
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda().squeeze(1)
    a = torch.sum(x ** 2, 1).unsqueeze(1)
    b = torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1)
    c = 2 * torch.matmul(x, y.transpose(0, 1))
    dist = a + b - c
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(gallery_data, probe_data , config):

    gallery_feature,  gallery_date, gallery_label = gallery_data[0], gallery_data[1], gallery_data[2]
    probe_feature, probe_date, probe_label = probe_data[0], probe_data[1], probe_data[2]

    gallery_feature = np.array(gallery_feature)
    probe_feature = np.array(probe_feature)

    dist = cuda_dist(probe_feature, gallery_feature)
    idx = dist.sort(1)[1].cpu().numpy()

    print("gallery len", len(gallery_label))

    num_rank = 5
    predict = list()
    # for i in range(len(probe_label)):
    #     predict.append(gallery_label[])

    pred_label = np.asarray([[gallery_label[idx[i][j]] for j in range(num_rank) ] for i in range(len(idx))])

    print(" pred label ",pred_label.shape)

    num_probe = len(pred_label)
    right_probe_top1 = 0
    right_probe_top5 = 0

    file = None

    if config.test.result_save:
        txt_path = os.path.join(config.train.dir, config.test.result_name)
        print(txt_path)
        file = open(txt_path, "a+")

    for i in range(num_probe):

        top1 = 0
        top5 = 0

        if probe_label[i] in pred_label[i][0:num_rank]:
            right_probe_top5 += 1
            top5 = 1

            if probe_label[i] == pred_label[i][0]:
                top1 = 1
                right_probe_top1 += 1

        if config.test.result_save:
            str_str = str(probe_date[i]) + "," + str(gallery_date[gallery_label.index(probe_label[i])]) + "," + str(probe_label[i]) + "," + str(top1) + "," + str(top5) + "\n"
            file.write(str_str)

    if file is not None:
        file.close()

    print("the top1 accuracy is : {}%, \nthe rank 5 accuracy is {}%. ".format(right_probe_top1*100.0/num_probe, right_probe_top5*100.0/num_probe))
    return right_probe_top1*100.0/num_probe, right_probe_top5*100.0/num_probe, num_probe


class Evaluator(object):
    def __init__(self, gallery_data, probe_data , config):
        self.num_rank = 5
        self.config = config
        self.gallery_feature,  self.gallery_date, self.gallery_label = gallery_data[0], gallery_data[1], gallery_data[2]
        self.probe_feature, self.probe_date, self.probe_label = probe_data[0], probe_data[1], probe_data[2]
        self.gallery_feature = np.array(self.gallery_feature)
        self.probe_feature = np.array(self.probe_feature)

        self.idx = list()
        print("gallery len", len(self.gallery_label))

        self.predict_lable = None


    def model_predict(self):
        if self.config.test.vote:
            self.idx = self.update_idx()
        return np.asarray([[self.gallery_label[self.idx[i][j]] for j in range(self.num_rank)] for i in range(len(self.idx))])


    def update_compute(self, video_index):
        # update specific list for idx according to video_index
        new_index = list()
        for i in range(len(video_index)):
            index_ = list()
            data =np.asarray(video_index[i])[:, 0:self.num_rank]

            # top1_temp = list(set[data[j][0] for j in range(len(data))])
            # top1_num = np.zeros(len(top1_temp)).tolist()

            temp = list()
            for n in range(len(data)):
                temp += data[n].tolist()

            top5_temp = list(set(temp))
            top5_num = np.zeros(len(top5_temp)).tolist()

            for kk in range(len(data)):                
                for ss in range(len(data[kk])):

                    if data[kk][ss] in top5_temp:
                        temp_index = top5_temp.index(data[kk][ss])
                        top5_num[temp_index] +=1
            
            index_.append(top5_temp)
            index_.append(top5_num)
            index_ = np.asarray(index_)
            index_ = index_.T[np.argsort(-index_[1,:])].T

            for j in range(len(data)):
                if len(new_index) == 0:
                    new_index = np.asarray(index_[0][0:self.num_rank])
                else:
                    new_index = np.vstack([new_index, index_[0][0:self.num_rank]])
        new_index = new_index.astype(np.int64)
        return new_index


    def update_idx(self):
        new_idx = None
        temp_date = None
        temp_label = None
        video_index = list()
        video_temp = list()
        for i in range(len(self.idx)):
            if i == 0:
                temp_date = self.probe_date[0]
                temp_label = self.probe_label[0]
                video_temp.append(self.idx[i,:])
            
            elif temp_date == self.probe_date[i] and temp_label == self.probe_label[i]:
                video_temp.append(self.idx[i,:])
            else: 
                temp_date = self.probe_date[i]
                temp_label = self.probe_label[i]
                video_index.append(video_temp)
                video_temp = list()
                video_temp.append(self.idx[i,:])

        video_index.append(video_temp)    # to add the last one in video_index

        new_idx = self.update_compute(video_index)
        print("len data = ", len(video_index))
        return new_idx

    
    def get_dist(self, dist):
        self.idx = dist

    def run(self):
        # pred_label = np.asarray([[gallery_label[idx[i][j]] for j in range(num_rank) ] for i in range(len(idx))])
        if len(self.idx) == 0:
            dist = cuda_dist(self.probe_feature, self.gallery_feature)
            self.idx = dist.sort(1)[1].cpu().numpy()
        self.predict_lable = self.model_predict()

        print(" pred label ", self.predict_lable.shape)

        num_probe = len(self.predict_lable)
        right_probe_top1 = 0
        right_probe_top5 = 0
        file =  None

        if self.config.test.result_save:
            txt_path = os.path.join(self.config.train.dir, self.config.test.result_name)
            print(txt_path)
            file = open(txt_path, "w+")

        for i in range(num_probe):

            top1 = 0
            top5 = 0

            if self.probe_label[i] in self.predict_lable[i][0:self.num_rank]:
                right_probe_top5 += 1
                top5 = 1

                if self.probe_label[i] == self.predict_lable[i][0]:
                    top1 = 1
                    right_probe_top1 += 1

            if self.config.test.result_save:
                str_str = str(self.probe_date[i]) + "," + str(self.gallery_date[self.gallery_label.index(self.probe_label[i])]) + "," + str(self.probe_label[i]) + "," + str(top1) + "," + str(top5) + "\n"
                file.write(str_str)

        if file is not None:
            file.close()

        print("the top1 accuracy is : {}%, \nthe rank 5 accuracy is {}%. ".format(right_probe_top1*100.0/num_probe, right_probe_top5*100.0/num_probe))
        return right_probe_top1*100.0/num_probe, right_probe_top5*100.0/num_probe, num_probe


def eval_voting(gallery_data, probe_data, config):

    gallery_feature,  gallery_date, gallery_label = gallery_data[0], gallery_data[1], gallery_data[2]
    probe_feature, probe_date, probe_label = probe_data[0], probe_data[1], probe_data[2]

    gallery_feature = np.array(gallery_feature)
    probe_feature = np.array(probe_feature)

    dist = cuda_dist(probe_feature, gallery_feature)
    idx = dist.sort(1)[1].cpu().numpy()

    print("gallery len", len(gallery_label))

    num_rank = 5
    predict = list()
    # for i in range(len(probe_label)):
    #     predict.append(gallery_label[])



    pred_label = np.asarray([[gallery_label[idx[i][j]] for j in range(num_rank) ] for i in range(len(idx))])

    print(" pred label ",pred_label.shape)

    num_probe = len(pred_label)
    right_probe_top1 = 0
    right_probe_top5 = 0

    file = None

    if config.test.result_save:
        txt_path = os.path.join(config.train.dir, config.test.result_name)
        print(txt_path)
        file = open(txt_path, "w")

    for i in range(num_probe):

        top1 = 0
        top5 = 0

        if probe_label[i] in pred_label[i][0:num_rank]:
            right_probe_top5 += 1
            top5 = 1

            if probe_label[i] == pred_label[i][0]:
                top1 = 1
                right_probe_top1 += 1

        if config.test.result_save:
            str_str = str(probe_date[i]) + "," + str(gallery_date[gallery_label.index(probe_label[i])]) + "," + str(probe_label[i]) + "," + str(top1) + "," + str(top5) + "\n"
            file.write(str_str)

    if file is not None:
        file.close()

    print("the top1 accuracy is : {}%, \nthe rank 5 accuracy is {}%. ".format(right_probe_top1*100.0/num_probe, right_probe_top5*100.0/num_probe))
    return right_probe_top1, right_probe_top5, num_probe
