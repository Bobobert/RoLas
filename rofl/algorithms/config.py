"""
    Template for the config dictionary to set an experiment.
"""
from rofl.functions.const import *

agent = {
    'id' : 0,
    'gamma' : GAMMA,
    'lambda' : LAMDA_GAE,
    'gae' : False,
}

train = {
    'epochs' : 10**2,
    'test_freq' : 10,
    'test_iters' : 20,
    'expected_perfomance' : None,
    'max_performance' : None,
    'max_time' : None,
    'max_steps_per_test' : -1,
}

policy = {
    'evaluate_freq' : 10,
    'evaluate_max_grad' : True,
    'evaluate_mean_grad' : True,
    'clip_grad' : 0,
}

env = {
    'name' : None,
    'warmup' : None,
    'warmup_min_steps' : 0,
    'warmup_max_steps' : 30,
    'obs_shape' : None,
    'obs_mode' : None,
    'max_length' : 10**3,
    'seedTrain' : TRAIN_SEED,
    'seedTest' : TEST_SEED,
}

config = {
    'env' : env,
    'agent' : agent,
    'policy' : policy,
    'train' : train,
    'variables' : [],
}

def completeDict(targetDict:dict, otherDict:dict):
    """
        Completes target with the keys of other, if they are not present.
        Does not use copy.
    """
    for key in otherDict.keys():
        target = targetDict.get(key)
        load = otherDict[key]
        if target is not None:
            if isinstance(target, dict) and isinstance(load, dict):
                targetDict[key] = completeDict(target, load)
        else:
            targetDict[key] = load
    return targetDict

def createConfig(*targetDicts:dict):
    newConfig = config.copy()
    for targetDict in targetDicts:
        newConfig = completeDict(targetDict, newConfig)
    return newConfig
