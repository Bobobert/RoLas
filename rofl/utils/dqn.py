from typing import Tuple
from rofl.functions.const import *
from rofl.functions.dicts import obsDict
from rofl.functions.functions import Tcat, newZero, no_grad, Tmean
from rofl.functions import runningStat
from rofl.functions.torch import array2Tensor

def genFrameStack(config):
    lhist, channels = config['agent']['lhist'], config['agent'].get('channels', 1)
    return np.zeros((lhist * channels, *config['env']['obs_shape']), dtype = np.uint8)

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

    if reset:
        framestack.fill(0)
    else:
        framestack = np.roll(framestack, 1, axis = 0)

    framestack[0] = obs
    agent.frameStack = framestack

    return array2Tensor(framestack, agent.device).div(255.0)

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

def composeLHistv0(frame: TENSOR, position: Tuple[int,int], time: float):
    frame = frame.flatten(1)
    extra = torch.tensor([*position, time], dtype = F_TDTYPE_DEFT, device = frame.device).unsqueeze_(1)
    return Tcat([frame, extra], dim = 1).detach_()

def decomposeLHistv0(observation, frameShape) -> Tuple[TENSOR, TENSOR]:
    """
        returns
        -------
        frames: Tensor
        pos: Tensor
    """
    time = observation[:,-1]
    position = observation[:,-3:-1]
    frames = observation[:,:-3].reshape(-1, frameShape)
    return frames, position, time
