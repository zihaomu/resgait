# Copyright 2018 pudae. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .initialization_new_label import get_initial, get_gallery_data, get_initial_test, get_initial_test_save, get_gallery_data_loader, get_initial_test_for_covariate, get_initial_test_for_covariate_pose
from .random import random_select, random_clip
from .collate_fns import get_collate_fn, collate_fn_clip_probe, update_gallery
from .config import load
from .evaluator import evaluation, Evaluator
from .time_split import small_day, big_day
from .cal_center import L2_distance, Vector_module, cuda_dist_tensor
from  .new_split import train_id, test_id