"""
    Constansts and functions from libraries to use in all the files.
"""
import os, sys
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random as rnd
import numpy as np
import numpy.random as nprnd

### FUNCTION FROM LIBS ###
ceil = math.ceil
floor = math.floor
Tsum = torch.sum
Tlog = torch.log
Tcat = torch.cat
Tmul = torch.mul
Tdiv = torch.div
Tpow = torch.pow
Tmean = torch.mean
Tstd = torch.std
Tdevice = torch.device
Texp = torch.exp
Tdot = torch.dot
Tsqrt = torch.sqrt
Tstack = torch.stack
no_grad = torch.no_grad
try:
    from numba.typed import List
except:
    print("Numba support not available. Errors could be heading one's way")
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from .dummy import dummyTBW as SummaryWriter

#### LITTLE USEFUL FUNCTIONS ###
def assertProb(sus):
    f = (sus >= 0.0) and (sus <= 1.0)
    if not f:
        raise ValueError("Value must be in [0,1]")
    return sus

def assertIntPos(sus):
    f = (sus > 0) and isinstance(sus, (int))
    if not f:
        raise ValueError("Value must be an integer greater than 0")
    return sus

def sqrConvDim(inpt,kernel,stride, pad = 0, dil = 1):
    return floor((inpt + 2*pad - dil*(kernel-1) - 1) /stride + 1)

def runningMean(xt, y, t):
    """
        parameters
        ----------
        xt: float
            mean
        y: int/float
            new value
        t: int
            observations included in xt
        
        returns
        -------
        xt1: float
            new mean
    """
    return xt * t / (t + 1) + y / (t + 1)

def mutiplyIter(itm):
    t = 1
    for i in itm:
        t *= i
    return t

from typing import Union

### DEFAULTS TYPES ###
DEVICE_DEFT = Tdevice("cpu")
F_TDTYPE_DEFT = torch.float32
I_TDTYPE_DEFT = torch.int64
F_NDTYPE_DEFT = np.float32
I_NDTYPE_DEFT = np.int32
B_TDTYPE_DEFT = torch.bool
B_NDTYPE_DEFT = np.bool_
TENSOR = torch.Tensor
ARRAY = np.ndarray

### CONSTANTS DEFAULTS ###
TEST_N_DEFT = 20
MAX_EPISODE_LENGTH = -1
OPTIMIZER_DEF = "adam"
OPTIMIZER_LR_DEF = 5e-5
MINIBATCH_SIZE = 32
PI = math.pi
PLATFORM = sys.platform
NCPUS = os.cpu_count() 
if PLATFORM == "win32":
    NCPUS += -1
TRAIN_SEED, TEST_SEED = 117, 404

### DQN ###
MEMORY_SIZE = 10**6
LHIST = 4
FRAME_SIZE = [84,84]
GAMMA = 0.99
LAMDA_GAE = 1.0
RNN_BOOT_DEFT = 10

### TRPO ###
MAX_DKL = 1e-2
CG_DAMPING = 1e-3

### PPO ###
ENTROPY_LOSS = 1e-2
EPS_SURROGATE = 0.1
PPO_EPOCHS = 10

### GX ###
ALPHA = 0.15
LINEWDT = 2
CLRRDM = "red"
CLRPI = "blue"