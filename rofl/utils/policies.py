from typing import Union

from numpy.lib.function_base import insert
from rofl.functions.const import ARRAY, F_TDTYPE_DEFT, TENSOR, torch

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

def logProb4Action(policy, observation:TENSOR, action:Union[TENSOR, ARRAY, int, float]) -> TENSOR:
    if isinstance(action, ARRAY):
        action = torch.from_numpy(action).unsqueeze(0)
    elif isinstance(action, (int, float)):
        action = torch.tensor([action], device = observation.device)
    actor = policy.actor
    params = actor.onlyActor(observation)
    print(f'logProb: action type {type(action)}, shape {action.shape}. Observation shape {observation.shape}')
    dist = actor.getDist(params)
    log_prob = dist.log_prob(action)
    return log_prob

def setEmptyOpt(policy):
    policy.tbw = None
    config = policy.config
    # set config for dummy optimizers
    config['policy']['network']['optimizer'] = 'dummy'
    if config['policy']['baseline']['networkClass'] is not None:
        config['policy']['baseline']['optimizer'] = 'dummy'

def calculateGAE(advantage: TENSOR, done: TENSOR, gamma: float, lmbd: float):
    gaeFactor, gaeRun = gamma * lmbd, 1.0
    gaes = advantage.new_zeros(advantage.shape)
    gaes[-1] = advantage[-1]
    for i in range(len(advantage) - 1, -1, -1):
        if done[i]:
            gaeRun = 1.0
            gaes[i] = advantage[i]
        else:
            gaeRun *= gaeFactor
            gaes[i] = advantage[i] + gaeRun * gaes[i + 1]
    gaes.detach_()
    return gaes
        