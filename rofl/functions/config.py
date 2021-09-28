"""
    Template for the config dictionary to set an experiment.
"""
def completeDict(targetDict:dict, otherDict:dict):
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

def createConfig(*targetDicts:dict, expName: str = 'unknown'):
    from rofl.functions.configDeft import config
    newConfig = config.copy()
    newConfig['algorithm'] = expName
    newConfig = completeDict(getConfigAlg(expName), newConfig)
    for targetDict in targetDicts:
        newConfig = completeDict(targetDict, newConfig)
    return newConfig

def getConfigAlg(expName):
    import rofl.algorithms as algs
    alg = getattr(algs, expName)
    return alg.algConfig

def getTrainFun(config):
    import rofl.algorithms as algs
    alg = alg = getattr(algs, config['algorithm'])
    return alg.train

def createAgent(config, policy, envMaker, **kwargs):
    import rofl.agents as agents
    agentClass = getattr(agents, config['agent']['agentClass'])
    return agentClass(config, policy, envMaker, **kwargs)

def createPolicy(config, **kwargs):
    import rofl.policies as policies
    c = config['policy']
    policyClass = getattr(policies, config['policy']['policyClass'])
    policy = policyClass(config, **kwargs)
    return policy

def getEnvMaker(config):
    import rofl.envs as envs
    envMaker = getattr(envs, config['env']['envMaker'])
    return envMaker(config)

def createActor(config:dict, key:str = 'network'):
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
    netClass = getattr(nets, config['policy'][key]['networkClass'])
    return netClass(config)
