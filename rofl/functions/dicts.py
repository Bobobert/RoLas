"""
    Functions to manage the information dictionaries, which have the goal to pass
    experiences from an envVec or agent to another agent or policy.

    A batch or info dictionary should be have the next keys:
        N - number of items or the batch size contained
        device - NoneType or torch.device

    For every other key, if is a list, List, np.array, or torch.tensor is considered to
    be stacked in the main dict. Any other type is stack on a list.

"""

from typing import Tuple
from numpy import dtype
from .functions import combDeviations, torch
from .const import *
import time

def obsDict(obs, nextObs, action, reward, step, done, info = {}, **kwargs) -> dict:
    
    zeroDevice = obs.device if isinstance(obs, TENSOR) else DEVICE_DEFT

    dict_ = {"observation": obs, "device": zeroDevice,
            "next_observation" : nextObs,
            "action": action, "reward": reward, 
            "step": step, "done": done, 
            "info": info, "N": 1,
            }

    for k in kwargs.keys():
        dict_[k] = kwargs[k]

    return dict_

def initResultDict():
    return {"mean_return": [],
            "mean_steps": [],
            "std_return": [],
            "std_steps": [],
            "custom": [],
            "time_start": time.time(),
            "time_elapsed": 0.0,
            "time_execution": [],
            "max_return": [],
            "min_return": [],
            "tot_tests": []
            }

def mergeDicts(*batchDicts, targetDevice = DEVICE_DEFT, keys = None):
    """
        Function to merge info dicts (preferably of the same origin) into
        a single big dict.

        parameters
        ----------
        batchDicts: dict
            Source info for the new bigger batch
        targetDevice: torch.device
            Default 'cpu'
        keys: list of str
            Default None, uses all the keys available. If given a list
            or tuple of strings from the dictionaries, the batch can be reduced
            to those.
    """
    # Making work if more than one dict has came, else return the dict
    if len(batchDicts) > 1:
        # Construct the manager dict
        zero = batchDicts[1] 
        templateDict, dtypes, shapes, zeroDevice = {}, {}, {}, zero["device"]
        if keys is None:
            keys = zero.keys()
        for k in keys:
            if k == 'N' or k == 'device':
                continue
            templateDict[k] = None
            item = zero[k]
            shapes[k] = item.shape if hasattr(item, 'shape') else None
            dtypes[k] = item.dtype if hasattr(item, 'dtype') else None

        # Collecting N, checking any device dispair
        N, allSameDev = 0, True
        for d in batchDicts:
            N += d["N"]
            allSameDev *= d["device"] == zeroDevice
        zeroDevice = zeroDevice if allSameDev else DEVICE_DEFT

        # Complete manager dict
        for k in templateDict.keys():
            type_ = type(zero[k])
            dtype, shape = dtypes[k], shapes[k]
            if type_ == ARRAY:
                new = np.zeros((N, *shape), dtype = dtype)
            elif type_ == TENSOR:
                # TODO: check this behavior, is expecting any tensor come in batch form
                # while array is not
                new = torch.zeros((N, *shape[1:]), dtype = dtype, device = zeroDevice)
            elif type_ == List:
                new = List()
            else:
                new = list()
            templateDict[k] = new

        # Iterate from the dicts
        m = 0
        for d in batchDicts:
            n = d["N"]
            for k in templateDict.keys():
                ref, target = templateDict[k], d.get(k)
                if isinstance(target, ARRAY):
                    ref[m:m+n] = target
                elif isinstance(target, TENSOR):
                    ref[m:m+n] = target.to(zeroDevice) if not allSameDev else target
                elif isinstance(target, list):
                    ref += target
                elif isinstance(target, List):
                    for i in target:
                        ref.append(i)
                else:
                    try:
                        ref.append(target)
                    except AttributeError:
                        print(f'Target was {type(target)} while ref is a {type(ref)}. Using key {k} with dict: {d}')
            m += n
        assert N == m, 'Houston, something went wrong with this batch! expected %d samples but got %d' % (N, m)
        templateDict["device"] = zeroDevice
        templateDict["N"] = N
        return dev2devDict(templateDict, targetDevice)
    else:
        return dev2devDict(batchDicts[0], targetDevice)
        
def dev2devDict(infoDict: dict, targetDevice):
    if infoDict["device"] == targetDevice:
        return infoDict
    for k in infoDict.keys():
        target = infoDict[k]
        if isinstance(target, TENSOR):
            infoDict[k] = target.to(device = targetDevice)
    infoDict["device"] = targetDevice
    return infoDict

def addBootstrapArg(obsDict: dict):
    obsDict['bootstrapping'] = 0.0

def mergeResults(*dicts):
    mR, sR, mS, sS, N = 0, 0, 0, 0, 0
    mC, maxR, minR = 0, -np.inf, np.inf
    for d in dicts:
        t = d['tot_tests']
        mR, sR = combDeviations(mR, d['mean_return'], N, t, sR, d['std_return'])
        mS, sS = combDeviations(mS, d['mean_steps'], N, t, sS, d['std_steps'])
        mC = d['custom']
        N += t
        maxD, minD = d['max_return'], d['min_return']
        if maxD > maxR:
            maxR = maxD
        if minD < minR:
            minR = minD

    results = {
        'mean_return': mR,
        'std_return': sR,
        'mean_steps': mS,
        'std_steps': sS,
        'custom': mC,
        'max_return': maxR,
        'min_return': minR,
        'tot_tests' : N,
    }

    return results

def composeObs(*infoDicts, device = DEVICE_DEFT) -> Tuple[TENSOR, TENSOR, list]:
    # a hard-coded mergeDicts
    ids = []
    for i, dict_ in enumerate(infoDicts):
        obs = dict_['observation']

        if i == 0:# init tensors
            l = len(infoDicts)
            newObs = obs.new_zeros((l, *obs.shape[1:]), device = device)
            dones = obs.new_zeros((l,), dtype = B_TDTYPE_DEFT, device = device)
        
        ids.append(dict_['id'])
        newObs[i] = obs.to(device)
        dones[i] = dict_['done']

    return newObs, dones, ids

def solveOthers(other, ids, otherKey, *infoDicts):
    iDs = {}
    for dict_ in infoDicts:
        iDs[dict_['id']] = dict_
    for other, iD in zip(other, ids):
        dict_ = iDs[iD]
        if isinstance(other,TENSOR): # Return to batch notation
            other = other.unsqueeze(0)
        dict_[otherKey] = other
    