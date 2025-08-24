from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
from torchvision import models

# DATASET 
TRAIN_DIR = "path/to/train"
VAL_DIR = "path/to/val"
TEST_DIR = "path/to/test"


# save best model and history
BEST_MODEL_PATH = "model_name.pth"
MODEL_HISTORY_PATH = "history_name.pkl"


# model specific global variables
HYBRID = {
    "vit": "vit_b_16", # vit model you want
    "resnet": "resnet18" # resnet model you want
}

# Pretrained model
VIT_NAME2MODEL = {
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
}

RESNET_NAME2MODEL = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}
PRETRAINED = True
PROGRESS = True

# optimizer
LR = 1e-04
EPSILON = 1e-08
WEIGHT_DECAY = 1e-04
BETAS = (0.9, 0.999)

# model
NUM_CLASSES = 1
IMG_SIZE = 224
BATCH_SIZE = 5
N_EPOCHS = 2
MULTICLASSES = False # True = multi-class classification, False = binary

LABELS = {
    "folder name": "index" # example: The first folder APPLE stores images of apples. => "APPLE": 0 
}

Id2LABELS = {value: key for key, value in LABELS.items()}

NUM_WORKS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation config
AVERAGE = "micro" if MULTICLASSES else "binary"

