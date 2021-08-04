from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict


def _get_default_config():
  c = edict()
  c.if_train = True
  c.WORK_PATH = "./results"
  c.CUDA_VISIBLE_DEVICES = "0"
  c.writer = None # "PATH_TO_YOUR_TENSORLOG_DIR"  # None
  c.writer_name = 'data/'
  c.time_span = False

  # dataset
  c.data = edict()
  c.data.dataset = "OURS"  # OURS is for ReSGait, or OUMVLP
  c.data.name = 'silhouette' # pose, silouette, GEI
  c.data.dir = 'PATH_TO_YOUR_dataset'          # dataset
  c.data.cache = True
  c.data.cache_path = 'PATH_TO_SAVE_THE_CACHE_FILE'
  c.data.pid_num = 86
  c.data.random_seed = 999
  c.data.num_classes = 86
  c.data.pid_shuffle = False
  c.data.resolution = 64
  c.data.frame_num = 1
  c.data.num_workers = 4
  c.data.drop_last = False
  c.data.collate_fn = "clip"  # for pose data, it muse be set as None
  c.data.sampler = 'batch' # options: batch and weight
  c.data.appendix = None
  
  # model
  c.model = edict()
  c.model.name = 'SilhouetteNormal' # MORE model can be found at "./models/"
  c.model.params = edict()

  # train
  c.train = edict()
  c.train.finetuning = None
  c.train.weight = 1
  c.train.dir = './results'      # model save path
  c.train.restore_iter = 0          # training checkpoint       
  c.train.num_epochs = 8000
  c.train.num_grad_acc = None
  c.train.center_wight = 1
  c.train.batch_size = edict() 
  c.train.validation = False

  c.test = edict()
  c.test.epoch = 9
  c.test.evaluator = 'l2'   #
  c.test.gallery_model = "fix"
  c.test.result_save = True
  c.test.sampler = 'seq'
  c.test.result_name = 'result_seq.txt'
  c.test.covariate = "normal"
  c.test.vote = True 

  # optimizer
  c.optimizer = edict()
  c.optimizer.weight_decay = 0
  c.optimizer.name = 'adam' # adam, adam_sgd-> softmax+center
  c.optimizer.params = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'step' # step, multistep, exponential
  c.scheduler.base_lr = 1e-4
  c.scheduler.max_lr = 5e-4
  c.scheduler.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = 'cross_entropy' # cross_entropy, softmax_center
  c.loss.params = edict()

  return c

def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v

def load(config_path):
  with open(config_path, 'r') as fid:
    yaml_config = edict(yaml.load(fid))

  config = _get_default_config()
  _merge_config(yaml_config, config)

  return config
