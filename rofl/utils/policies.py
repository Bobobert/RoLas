from typing import Union
from rofl.functions.const import ARRAY, F_TDTYPE_DEFT, TENSOR
from rofl.functions.functions import Tmul, isBatch, newZero, torch, no_grad
from rofl.functions.torch import clipGrads

def getParamsBaseline(policy, observations):
    if policy.baseline is None:
        baselines = torch.zeros((observations.shape[0], 1), device = observations.device, dtype = F_TDTYPE_DEFT)
    elif not policy.actorHasCritic:
        baselines = policy.baseline(observations)
        params = policy.actor.onlyActor(observations)
    else:
        baselines, params = policy.actor(observations)
    params = params if policy.actorHasCritic else policy.actor.onlyActor(observations)
    return params, baselines

def getBaselines(policy, observations):
    if policy.baseline is None:
        baselines = torch.zeros((observations.shape[0], 1), device = observations.device, dtype = F_TDTYPE_DEFT)
    elif not policy.actorHasCritic:
        baselines = policy.baseline(observations)
    else:
        baselines = policy.actor.onlyValue(observations)
    return baselines

def actorSampleAction(actor, params):
    dist = actor.getDist(params)
    action = dist.sample()
    logProb = dist.log_prob(action)
    action = actor.processAction(action)
    return action, logProb

def getActionWProb(actor, observation):
    """
        Combined method

        returns
        --------
        - action
        - log_prob for action
    """
    params = actor.onlyActor(observation)
    return actorSampleAction(actor, params)

def getActionWValProb(actorCritic, observation):
    value, params = actorCritic(observation)
    action, logProb = actorSampleAction(actorCritic, params)
    return action, value, logProb

def logProb4Action(policy, observation:TENSOR, action:Union[TENSOR, ARRAY, int, float]) -> TENSOR:
    if isinstance(action, ARRAY):
        action = torch.from_numpy(action).unsqueeze(0)
    elif isinstance(action, (int, float)):
        action = torch.tensor([action], device = observation.device)
    actor = policy.actor
    params = actor.onlyActor(observation)
    dist = actor.getDist(params)
    action = actor.unprocessAction(action, isBatch(observation))
    log_prob = dist.log_prob(action)
    return log_prob

def setEmptyOpt(policy):
    policy.tbw = None
    config = policy.config
    # set config for dummy optimizers
    config['policy']['network']['optimizer'] = 'dummy'
    if config['policy']['baseline']['networkClass'] is not None:
        config['policy']['baseline']['optimizer'] = 'dummy'

def calculateReturn(policy, nextObsevation, dones, rewards):
    with no_grad():
        valueST1 = getBaselines(policy, nextObsevation[-1].unsqueeze(0))
    returns = newZero(rewards).cpu()
    lastReturn = valueST1[0]
    gamma = policy.gamma
    for i in range(rewards.shape[0], -1, -1):
        if dones[i]:
            lastReturn = 0.0
        returns[i] = lastReturn = rewards[i] + gamma * lastReturn
    return returns

def genMiniBatchLin(miniBatchSize, batchSize, *targets):
    for lower in range(0, batchSize, miniBatchSize):
        newYield = []
        for t in targets:
            upper = lower + miniBatchSize
            if upper > batchSize:
                upper = batchSize
            newYield.append(t[lower:upper])
        yield newYield

def calculateGAE(policy,  valuesST, nextObservations, dones, rewards, gamma, lmbda) -> TENSOR:
    with no_grad():
        valuesST1 = getBaselines(policy, nextObservations)
    # calculate TDs
    notDones = dones.bitwise_not()
    valuesST1 = Tmul(valuesST1, notDones)
    tds = rewards + gamma * valuesST1 - valuesST

    gaeFactor, gaeRun = gamma * lmbda, 1.0
    gaes = rewards.new_zeros(rewards.shape)
    gaes[-1] = tds[-1]
    for i in range(len(rewards) - 2, -1, -1):
        if dones[i]:
            gaeRun = 1.0
            gaes[i] = tds[i]
        else:
            gaeRun *= gaeFactor
            gaes[i] = tds[i] + gaeRun * gaes[i + 1]
    gaes.detach_()

    return gaes

def trainBaseline(policy, baselines, returns, f) -> TENSOR:
    '''
        Policies with baseline atribute (which should be different 
        from Actor w Critic) can do a single step of train with this
        piece fo code. Design mainly for inheritance from pgPolicy

    '''
    lossBaseline = f(baselines, returns)
    optimizer = policy.optimizerBl
    optimizer.zero_grad()
    if policy.clipGrad > 0:
        clipGrads(policy.baseline, policy.clipGrad)
    lossBaseline.backward()
    optimizer.step()
    
    return lossBaseline
