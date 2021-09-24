"""
    General propuse libraries and some functions
"""

### IMPORTS ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random as rnd
import numpy as np
import numpy.random as nprnd
import numba as nb

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
Texp = torch.exp
Tdot = torch.dot
Tsqrt = torch.sqrt
Tstack = torch.stack
no_grad = torch.no_grad

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

def sqrConvDim(inpt, kernel, stride, pad = 0, dil = 1):
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

def multiplyIter(itm):
    result_ = 1
    for i in itm:
        result_ *= i
    return result_
