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

def obsDict(obs, action, reward, step, done, info = {}, n = 1, **kwargs) -> dict:
    
    zeroDevice = obs.device if isinstance(obs, TENSOR) else DEVICE_DEFT

    dict_ = {"observation": obs, "device": zeroDevice,
            "action": action, "reward": reward, 
            "step": step, "done": done, 
            "info": info, "N": n,
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
            templateDict[k] = None
            try:
                shapes[k] = zero[k].shape
                dtypes[k] = zero[k].dtype
            except AttributeError:
                shapes[k] = None
                dtypes[k] = None
            
        # Collecting N, checking ir device dispair
        N, allSameDev = 0, True
        for d in batchDicts:
            N += d["N"]
            allSameDev *= d["device"] == zeroDevice

        # Complete manager dict
        for k in templateDict.keys():
            type_ = type(zero[k])
            dtype, shape = dtypes[k], shapes[k]
            if type_ == ARRAY:
                new = np.zeros((N, *shape), dtype = dtype)
            elif type_ == TENSOR:
                new = torch.zeros((N, *shape), dtype = dtype,
                                      device = zeroDevice if allSameDev else DEVICE_DEFT)
            elif type_ == List:
                new = List()
            else:
                new = list()
            templateDict[k] = new

        # Iterate from the dicts
        m = 0
        for d in batchDicts:
            n = d["N"]
            for k in d.keys():
                ref, target= templateDict[k], d[k]
                if isinstance(target, (TENSOR, ARRAY)):
                    ref[m:m+n] = target # TODO; check how it behaves
                elif isinstance(target, List):
                    for i in target:
                        ref.append(i)
                elif isinstance(target, list):
                    ref += target
                else:
                    ref.append(target)
            m += n
        templateDict["device"] = zeroDevice if allSameDev else DEVICE_DEFT
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
