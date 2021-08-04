# If you want to save the ROC_EER figure, please save the predicte result first.

import os

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import numpy as np

root = "ROOT_PATH"
gender_temp = "gender.npy"
label_temp = "label.npy"

gender_data = np.load(os.path.join(root, gender_temp))
label_data = np.load(os.path.join(root, label_temp))

fpr, tpr, threshold = roc_curve(label_data, gender_data, pos_label=0)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, threshold)(eer)

print(eer)

image_path = os.path.join(root, "roc_video_avg.jpg")
roc_auc=auc(fpr, tpr)
plt.title('ROC with averaging product')
plt.plot(fpr, tpr,'b',label='AUC = %0.4f EER=%0.4f'%(roc_auc,eer))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(image_path)
# fnr = 1 - tpr
# eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
