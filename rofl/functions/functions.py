"""
    General propuse libraries and some functions
"""

from typing import Union
from copy import deepcopy
import math

import numpy as np
import numpy.random as nprnd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rnd
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
noGrad = torch.no_grad
Trelu = F.relu
Tsoftmax = F.softmax

#### LITTLE USEFUL FUNCTIONS ###
def assertProb(sus: Union[float, int]) -> Union[float, int]:
    f = (sus >= 0.0) and (sus <= 1.0)
    if not f:
        raise ValueError("Value must be in [0,1]")
    return sus

def assertIntPos(sus):
    f = (sus > 0) and isinstance(sus, (int))
    if not f:
        raise ValueError("Value must be an integer greater than 0")
    return sus

def sqrConvDim(inpt, kernel, stride, pad = 1, dil = 1):
    return floor((inpt + 2*pad - dil*(kernel-1) - 1) /stride + 1)

def isItem(T):
    if T.squeeze().shape == ():
        return True
    return False

def isBatch(T):
    shape = T.shape
    if shape == () or shape[0] <= 1:
        return False
    return True

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

def clipReward(agent, reward: Union[float, int, torch.Tensor]):    
    clipTarget = agent.clipReward
    if isinstance(reward, torch.Tensor):
        return torch.clamp(reward, min=-clipTarget, max=clipTarget)
    if clipTarget > 0:
        mClipTarget = -clipTarget
        if reward > clipTarget:
            reward = clipTarget
        elif reward < mClipTarget:
            reward = mClipTarget
    return reward

def isTerminalAtari(agent, info):
    done, lives = False, info.get('ale.lives', 0)
    if agent._reseted:
        agent.lives = lives
    elif lives != agent.lives:
        agent.lives = lives
        done = True # marked as terminal but no reset required
    return done

def inputFromGymSpace(config: dict) -> int:
    return multiplyIter(config['env']['observation_space'].shape)

def outputFromGymSpace(config: dict) -> int:
    aS = config['env']['action_space']
    if hasattr(aS, 'n'): # probing Discrete
        return getattr(aS, 'n')
    return multiplyIter(aS.shape) # Assuming Box

def reduceBatch(batch, op = Tsum):
    shape = batch.shape
    l = len(shape)
    if l < 2:
        return batch
    if l == 2 and shape[1] == 1:
        return batch.squeeze()
    dims = [n for n, _ in enumerate(shape[1:], 1)]
    return op(batch, dim=dims)

def combDeviations(m1, m2, n1, n2, s1, s2):
    """
        From: https://handbook-5-1.cochrane.org/chapter_7/table_7_7_a_formulae_for_combining_groups.htm
    """
    Ns = n1 + n2
    newMean = (m1 * n1 + m2 * n2) / Ns
    aux1 = (n1 - 1) * s1**2
    aux2 = (n2 - 1) * s2**2
    aux3 = (m1**2 + m2**2 -2 * m1 * m2) * n1 * n2 / Ns
    newStd = math.sqrt((aux1 + aux2 + aux3) / (Ns - 1))
    return newMean, newStd

def newZero(t, grad: bool = False):
    if isinstance(t, torch.Tensor):
        return t.new_zeros(t.shape, requires_grad=grad)
    elif isinstance(t, np.ndarray):
        return np.zeros_like(t)
    elif isinstance(t, tuple):
        zeros = []
        for i in t:
            zeros.append(newZero(i, grad))
        return tuple(zeros)
    else:
        raise NotImplementedError('t is %s which is not expected' % type(t))
