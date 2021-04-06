"""
    Constansts and functions from libraries to use in all the files.
"""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.device as Tdevice
import math
import random
import numpy as np

### FUNCTION FROM LIBS ###
ceil = math.ceil
floor = math.floor
Tcat = torch.cat
mul = torch.mul
no_grad = torch.no_grad

#### LITTLE USEFUL FUNCTIONS ###
def assertProb(sus):
    f = (sus >= 0.0) and (sus <= 1.0)
    if not f:
        raise ValueError("Value must be in [0,1]")

### DEFAULTS TYPES ###
DEVICE_DEFT = Tdevice("cpu")
F_TDTYPE_DEFT = torch.float32
I_TDTYPE_DEFT = torch.int64
F_NDTYPE_DEFT = np.float32
I_NDTYPE_DEFT = np.int32
TENSOR = torch.Tensor
ARRAY = np.ndarray

### CONSTANTS DEFAULTS ###
TEST_N_DEFT = 20
MAX_EPISODE_LENGTH = -1

### DQN ###
MEMORY_SIZE = 10**6
LHIST = 4
FRAME_SIZE = [84,84]