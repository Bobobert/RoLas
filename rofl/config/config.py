"""
    Template for the config dictionary to set an experiment.
"""
from typing import List, Tuple, Type, Union
from rofl.utils import Saver
from rofl.agents.base import Agent as AgentType
from rofl.policies.base import Policy as PolicyType
from rofl.networks.base import BaseNet, QValue, Value, Actor, ActorCritic
from gym import Env

def _evnMaker_(seed: int) -> Tuple[Env, List[int]]:
    pass

def _trainFun_(config: dict, agent: AgentType, policy:PolicyType, saver: Saver) -> dict: 
    pass

def completeDict(targetDict:dict, otherDict:dict) -> dict:
    """
        Completes target with the keys of other, if they are not present,
        stores the values of other. If they are, target keeps their values.
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

def createConfig(*targetDicts:dict, expName: str = 'unknown') -> dict:
    from rofl.config.defaults import config
    newConfig = config.copy()
    newConfig['algorithm'] = expName
    newConfig = completeDict(getConfigAlg(expName), newConfig)
    for targetDict in targetDicts:
        newConfig = completeDict(targetDict, newConfig)
    return newConfig

def getConfigAlg(expName) -> dict:
    import rofl.algorithms as algs
    alg = getattr(algs, expName)
    return alg.algConfig

def getTrainFun(config) -> _trainFun_:
    import rofl.algorithms as algs
    alg = alg = getattr(algs, config['algorithm'])
    return alg.train

def createAgent(config, policy, envMaker, **kwargs) -> AgentType:
    import rofl.agents as agents
    agentClass = getattr(agents, config['agent']['agentClass'])
    return agentClass(config, policy, envMaker, **kwargs)

def createPolicy(config, actor, **kwargs) -> PolicyType:
    import rofl.policies as policies
    c = config['policy']
    policyClass = getattr(policies, config['policy']['policyClass'])
    policy = policyClass(config, actor, **kwargs)
    return policy

def getEnvMaker(config) -> _evnMaker_:
    import rofl.envs as envs
    envMaker = getattr(envs, config['env']['envMaker'])
    return envMaker(config)

def createNetwork(config:dict, key:str = 'network') -> Union[BaseNet, Value, QValue, Actor, ActorCritic]:
    """
    Returns the desired network

    parameters
    ----------
    - config: dict
    - key: str
        'network', 'baseline' are options.
    """
    import rofl.networks
    nets = getattr(rofl.networks, config['algorithm'])
    if key == 'actor': key = 'network'
    netClass = getattr(nets, config['policy'][key]['networkClass'])
    return netClass(config)
