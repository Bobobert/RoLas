"""
    Functions to manage the information dictionaries, which have the goal to pass
    experiences from an envVec or agent to another agent or policy.

    A batch or info dictionary should be have the next keys:
        N - number of items or the batch size contained
        device - NoneType or torch.device

    For every other key, if is a list, List, np.array, or torch.tensor is considered to
    be stacked in the main dict. Any other type is stack on a list.

"""

from .const import *

def obsDict(obs, action, reward, step, done, info = {}, n = 1, **kwargs):
    
    dev = obs.device if isinstance(obs, TENSOR) else DEVICE_DEFT

    c = {"observation": obs, "device": dev,
            "action": action, "reward": reward, 
            "step": step, "done": done, 
            "info": info, "N":n,
            }

    for k in kwargs.keys():
        c[k] = kwargs[k]

    return c

def mergeDicts(*batchDicts, targetDevice = DEVICE_DEFT):
    # Making work if more than one dict has came, else return the dict
    if len(batchDicts) > 1:
        # Construct the manager dict
        zero = batchDicts[0]
        mngd, dtps, shps, dev = {}, {}, {}, zero["device"]
        for k in zero.keys():
            mngd[k] = None
            try:
                shps[k] = zero[k].shape
                dtps[k] = zero[k].dtype
            except AttributeError:
                shps[k] = None
                dtps[k] = None
            
        # Collecting N, checking ir device dispair
        N, allSameDev = 0, True
        for d in batchDicts:
            N += d["N"]
            allSameDev *= d["device"] == dev

        # Complete manager dict
        for k in mngd.keys():
            type_ = type(zero[k])
            dtype, shape = dtps[k], shps[k]
            if type_ == ARRAY:
                new = np.zeros((N, *shape[1:]), 
                                   dtype = dtype)
            elif type_ == TENSOR:
                new = torch.zeros((N, *shape[1:]), 
                                      dtype = dtype,
                                      device = dev if allSameDev else DEVICE_DEFT)
            elif type_ == List:
                new = List()
            else:
                new = list()
            mngd[k] = new

        # Iterate from the dicts
        m = 0
        for d in batchDicts:
            n = d["N"]
            for k in d.keys():
                ref, target= mngd[k], d[k]
                if isinstance(target, (TENSOR, ARRAY)):
                    ref[m:m+n] = target
                elif isinstance(target, List):
                    for i in target:
                        ref.append(i)
                elif isinstance(target, list):
                    ref += target
                else:
                    ref.append(target)
            m += n
        mngd["device"] = dev if allSameDev else DEVICE_DEFT
        mngd["N"] = N
        return dev2devDict(mngd, targetDevice)
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
