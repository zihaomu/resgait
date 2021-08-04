from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=W0611

from .silhouette import SilhouetteDeep, SilhouetteNormal, SilhouetteDeep_new_label, SilhouetteNormal_new_label, SilhouetteNormal_new_label8,SilhouetteDeep_new_label8, SilhouetteNormal_new_label_div, SilhouetteDeep_new_label_div
from .pose import PoseNet, PoseNet_no_normal, PoseNet_ft, PoseNet_new, CNN_pose_ft, CNN_pose_new
from .silhouette import Finetuning, SilhouetteNormal_new_label_ft

# models used in GEI experiments are same as in Silhouette experiments
# the GEI can be set as single frame video.
def get_GEINormal(frame_num, num_classes):
    model = SilhouetteNormal(num_classes=num_classes)
    return model

def get_CNN_pose_new(frame_num, num_classes):
    model = CNN_pose_ft(frame_num = frame_num, num_classes=num_classes)
    return model

def get_CNN_pose_ft(frame_num, num_classes):
    model = CNN_pose_new(frame_num = frame_num, num_classes=num_classes)
    return model

def get_GEIDeep(frame_num, num_classes):
    model = SilhouetteDeep(num_classes=num_classes)
    return model

def get_SilhouetteNormal(frame_num, num_classes):
    model = SilhouetteNormal(num_classes=num_classes)
    return model

def get_SilhouetteDeep(frame_num,num_classes):
    model = SilhouetteDeep(num_classes=num_classes)
    return model

def get_SilhouetteNormal_new_label(frame_num, num_classes):
    model = SilhouetteNormal_new_label(num_classes=num_classes)
    return model

# for finetuning model
def get_SilhouetteNormal_new_label_ft(frame_num, num_classes):
    print("model class number is ", num_classes)
    model = SilhouetteNormal_new_label_ft(num_classes=num_classes)
    return model

def get_SilhouetteNormal_new_label_div_4(frame_num, num_classes):
    model = SilhouetteNormal_new_label_div(num_classes=num_classes, div_channel= 4)
    return model

def get_SilhouetteNormal_new_label_div_1(frame_num, num_classes):
    model = SilhouetteNormal_new_label_div(num_classes=num_classes, div_channel= 1)
    return model

def get_SilhouetteNormal_new_label_div_1_4(frame_num, num_classes):
    model = SilhouetteNormal_new_label_div(num_classes=num_classes, div_channel= 1/4)
    return model

def get_SilhouetteNormal_new_label_div_2(frame_num, num_classes):
    model = SilhouetteNormal_new_label_div(num_classes=num_classes,div_channel= 2 )
    return model

def get_SilhouetteNormal_new_label8(frame_num, num_classes):
    model = SilhouetteNormal_new_label8(num_classes=num_classes)
    return model

def get_SilhouetteDeep_new_label(frame_num,num_classes):
    model = SilhouetteDeep_new_label(num_classes=num_classes)
    return model

def get_SilhouetteDeep_new_label_div_4(frame_num,num_classes):
    model = SilhouetteDeep_new_label_div(num_classes=num_classes, div_channel= 4)
    return model

def get_SilhouetteDeep_new_label_div_2(frame_num,num_classes):
    model = SilhouetteDeep_new_label_div(num_classes=num_classes, div_channel= 2)
    return model

def get_SilhouetteDeep_new_label8(frame_num,num_classes):
    model = SilhouetteDeep_new_label8(num_classes=num_classes)
    return model

def get_PoseNet(frame_num, num_classes):
    model = PoseNet(frame_num , num_classes)
    return model

def get_PoseNet_new(frame_num, num_classes):
    model = PoseNet_new(frame_num , num_classes)
    return model

# for finetuning model
def get_PoseNet_ft(frame_num, num_classes):
    model = PoseNet_ft(frame_num , num_classes)
    return model

def get_PoseNet_no_normal(frame_num,num_classes):
    model = PoseNet_no_normal(frame_num, num_classes)
    return model

def get_Finetuning(config, model):
    model = Finetuning(model=model)
    return model

def get_model(config, model_old=None):
    model_name = config.model.name
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    if model_old is None:
        model = f(config.data.frame_num, config.data.num_classes)
    else:
        f = globals().get('get_Finetuning')
        model = f(config.data.num_classes, model_old)
    return model

def get_model_test(model_name):
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    return f()
    
if __name__ == '__main__':
    model_name = "SilhouetteNormal"
    f = get_model_test(model_name)
    print(f)
    #print(f())
