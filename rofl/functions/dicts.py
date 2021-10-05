"""
    Functions to manage the information dictionaries, which have the goal to pass
    experiences from an envVec or agent to another agent or policy.

    A batch or info dictionary should be have the next keys:
        N - number of items or the batch size contained
        device - NoneType or torch.device

    For every other key, if is a list, List, np.array, or torch.tensor is considered to
    be stacked in the main dict. Any other type is stack on a list.

"""

from rofl.functions.functions import combDeviations
from .const import *

def obsDict(obs, action, reward, step, done, info = {}, **kwargs) -> dict:
    
    zeroDevice = obs.device if isinstance(obs, TENSOR) else DEVICE_DEFT

    dict_ = {"observation": obs, "device": zeroDevice,
            "action": action, "reward": reward, 
            "step": step, "done": done, 
            "info": info, "N": 1,
            }

    for k in kwargs.keys():
        dict_[k] = kwargs[k]

    return dict_

def mergeDicts(*batchDicts, targetDevice = DEVICE_DEFT):
    # Making work if more than one dict has came, else return the dict
    if len(batchDicts) > 1:
        # Construct the manager dict
        zero = batchDicts[1] 
        templateDict, dtypes, shapes, zeroDevice = {}, {}, {}, zero["device"]
        for k in zero.keys():
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
                ref, target = templateDict[k], d[k]
                if isinstance(target, ARRAY):
                    ref[m:m+n] = target
                elif isinstance(target, TENSOR):
                    target.to(zeroDevice) if not allSameDev else None
                    ref[m:m+n] = target
                elif isinstance(target, List):
                    for i in target:
                        ref.append(i)
                elif isinstance(target, list):
                    ref += target
                else:
                    ref.append(target)
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
    obsDict['advantage'] = 0.0
    obsDict['bootstrapping'] = 0.0
    return obsDict

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
    