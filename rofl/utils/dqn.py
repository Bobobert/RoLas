from typing import Tuple
from rofl.functions.const import *
from rofl.functions.functions import newZero, no_grad, Tmean
from rofl.functions import runningStat
from rofl.functions.torch import array2Tensor
from rofl.utils.bulldozer import composeObsWContextv0, decomposeMultiDiscrete

def genFrameStack(config):
    lhist, channels = config['agent']['lhist'], config['agent'].get('channels', 1)
    if channels == 4:
        dtype = B_NDTYPE_DEFT
    else:
        dtype = UI_NDTYPE_DEFT
    return np.zeros((lhist * channels, *config['env']['obs_shape']), dtype = dtype)

def lHistObsProcess(agent, obs, reset):
    """
        From an already processed observation image type, modifies the
        lhist framestack.
        Agent is supossed to have a frameStack atribute.
    """
    try:
        framestack = agent.frameStack
    except AttributeError:
        agent.frameStack = framestack = genFrameStack(agent.config)
        #print("Warning: Agent didn't had frameStack declared") # TODO: add debuger level

    channels = agent.channels
    if reset:
        framestack.fill(0)
    else:
        framestack = np.roll(framestack, channels, axis = 0)

    framestack[0:channels] = obs
    agent.frameStack = framestack

    obsTensor = array2Tensor(framestack, agent.device)

    if channels == 1:
        return obsTensor.div(255.0)
    else:
        return obsTensor

def prepare4Ratio(agent):
    agent.ratioTree = runningStat()

def calRatio(agent, env):
    # Calculate ratio from environment
    cc = env.cell_counts
    tot = env.n_col * env.n_row
    agent.ratioTree += cc[env.tree] / tot

def reportQmean(agent):
    if agent.fixedTrajectory is None:
        return 0.0
    with no_grad():
        model_out = agent.policy.actor(agent.fixedTrajectory)
        mean = Tmean(model_out.max(1).values).item()
    if agent.tbw != None:
        agent.tbw.add_scalar("test/mean max Q", mean, agent.testCalls)
    return mean

def reportRatio(agent):
    meanQ = reportQmean(agent)
    if agent.tbw != None:
        agent.tbw.add_scalar("test/mean tree ratio", agent.ratioTree.mean, agent.testCalls)
        agent.tbw.add_scalar("test/std tree ratio", agent.ratioTree.std, agent.testCalls)
    return {"mean_q": meanQ, 
            "mean tree ratio": agent.ratioTree.mean, 
            "std tree ratio":agent.ratioTree.std}

def dqnStepv0(fun):
    """
        Manages the grid observations for the returned obsDict from
        a envStep or fullStep.

        Creates the zeroFrame.
    """
    def step(*args, **kwargs):
        obsDict = fun(*args, **kwargs)
        agent = args[0]
        prevFrame, lastFrame = agent.prevFrame, agent.lastFrame
        if prevFrame is None: # just needed once
            prevFrame = agent.prevFrame = agent.zeroFrame = newZero(lastFrame)
        # sligth mod to the obsDict from DQN memory
        obsDict['observation'] = prevFrame
        obsDict['next_observation'] = lastFrame
        obsDict['device'] = DEVICE_DEFT
        return obsDict
    return step

def processBatchv0(infoDict: dict) -> dict:
    infoDict['observation'] = infoDict['observation'].div(255.0).detach_()
    infoDict['next_observation'] = infoDict['next_observation'].div(255.0).detach_()
    return infoDict

def dqnStepv1(fun):
    """
        Manages the grid observations and context for the returned obsDict from
        a envStep or fullStep.

        Creates the zeroFrame and zeroContext attributes.
    """
    def step(*args, **kwargs):
        obsDict = fun(*args, **kwargs)
        agent = args[0]
        prevFrame, lastFrame = agent.prevFrame, agent.lastFrame
        # just needed once
        if prevFrame is None: 
            prevFrame = agent.prevFrame = agent.zeroFrame = newZero(lastFrame)
        prevContext, lastContext = agent.prevContext, agent.lastContext
        if prevContext is None:
            prevContext = agent.prevContext = agent.zeroContext = newZero(lastContext)
        obsDict['observation'] = prevFrame
        obsDict['next_observation'] = lastFrame
        # Saving parts of context, this will result in losing the kernel context from
        # memory, which also can lead to save memory if not needed
        obsDict['context_pos'] = prevContext[1]
        obsDict['context_time'] = prevContext[2]
        obsDict['next_context_pos'] = lastContext[1]
        obsDict['next_context_time'] = lastContext[2]
        obsDict['device'] = DEVICE_DEFT
        return obsDict
    return step

def processBatchv1(infoDict: dict, useChannels: bool, actionSpace) -> dict:
    observations, nObservations = infoDict['observation'], infoDict['next_observation']

    if not useChannels:
        observations = observations.div(255.0)
        nObservations = nObservations.div(255.0)

    contexts = (None, infoDict['context_pos'], infoDict['context_time'])
    nContext = (None, infoDict['next_context_pos'], infoDict['next_context_time'])

    infoDict['observation'] = composeObsWContextv0(observations, contexts, True)
    infoDict['next_observation'] = composeObsWContextv0(nObservations, nContext, True)
    infoDict['action'] = decomposeMultiDiscrete(infoDict['action'], actionSpace, True)
    return infoDict
