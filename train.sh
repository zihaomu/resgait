#!/bin/bash

python3 train_silhouette.py \
  --config=configs/silhouette.yml \
  --GPU_num=0,1

python3 train_pose.py \
  --config=configs/pose.yml \
  --GPU_num=0
