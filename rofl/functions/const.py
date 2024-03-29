"""
    Constants to use in all the files. Side effect of from .THIS import * is to 
    have torch, numpy (as np), math, os, sys imported as well.
"""
import os, sys
from .functions import torch, np, math
from numba.typed import List

### DEFAULTS TYPES ###
DEVICE_DEFT = torch.device("cpu")
F_TDTYPE_DEFT = torch.float32
I_TDTYPE_DEFT = torch.int64
UI_NDTYPE_DEFT = np.uint8
F_NDTYPE_DEFT = np.float32
I_NDTYPE_DEFT = np.int32
B_TDTYPE_DEFT = torch.bool
B_NDTYPE_DEFT = np.bool_
TENSOR = torch.Tensor
TDIST = torch.distributions.distribution.Distribution
ARRAY = np.ndarray
Tdevice = torch.device

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
DEFT_MEMORY_SIZE = 5*10**3
EPSILON_OP = 1e-5
DEFT_KEYS = ['observation', 'next_observation', 'action', 'done', 'reward']

### DQN ###
MEMORY_SIZE = 10**5
LHIST = 4
FRAME_SIZE = [84,84]
GAMMA = 0.99
LAMDA_GAE = 0.9
RNN_BOOT_DEFT = 10

### TRPO ###
MAX_DKL = 1e-2
CG_DAMPING = 1e-3

### PPO ###
ENTROPY_LOSS = 1e-2
EPS_SURROGATE = 0.1
PPO_EPOCHS = 10
LOSS_POLICY_CONST = 1.0
LOSS_VALUES_CONST = 0.6

### GX ###
ALPHA = 0.15
LINEWDT = 2
CLRRDM = "red"
CLRPI = "blue"
