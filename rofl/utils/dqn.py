from typing import Tuple
from rofl.functions.const import *
from rofl.functions.dicts import obsDict
from rofl.functions.functions import Tcat, newZero, no_grad, Tmean
from rofl.functions import runningStat
from rofl.functions.torch import array2Tensor

def genFrameStack(config):
    return np.zeros((config['agent']['lhist'], *config['env']['obs_shape']), dtype = np.uint8)

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
        print("Warning: obs invalid, agent didn't had frameStack declared") # TODO: add debuger level

    if reset:
        agent.lastFrameStack = newZero(framestack)
        framestack = newZero(framestack)
    else:
        agent.lastFrameStack = framestack
        framestack = np.roll(framestack, 1, axis = 0) # generates new array
    framestack[0] = obs
    agent.frameStack = framestack

    return array2Tensor(framestack, agent.device).div(255)

def prepare4Ratio(obj):
    obj.ratioTree = runningStat()

def calRatio(obj, env):
    # Calculate ratio from environment
    cc = env.cell_counts
    tot = env.n_col * env.n_row
    obj.ratioTree += cc[env.tree] / tot

def reportQmean(obj):
    if obj.fixedTrajectory is None:
        return 0.0
    with no_grad():
        model_out = obj.policy.dqnOnline(obj.fixedTrajectory)
        mean = Tmean(model_out.max(1).values).item()
    if obj.tbw != None:
        obj.tbw.add_scalar("test/mean max Q", mean, obj.testCalls)
    return mean

def reportRatio(obj):
    meanQ = reportQmean(obj)
    if obj.tbw != None:
        obj.tbw.add_scalar("test/mean tree ratio", obj.ratioTree.mean, obj.testCalls)
        obj.tbw.add_scalar("test/std tree ratio", obj.ratioTree.std, obj.testCalls)
    return {"mean_q": meanQ, 
            "mean tree ratio": obj.ratioTree.mean, 
            "std tree ratio":obj.ratioTree.std}

def dqnStep(fun):
    def step(*args, **kwargs):
        obsDict = fun(*args, **kwargs)
        agent = args[0]
        obsDict['observation'] = agent.lastFrameStack # omits the Tensor
        obsDict['next_observation'] = agent.frameStack
        obsDict['device'] = DEVICE_DEFT
        return obsDict
    return step

def processBatchv0(infoDict: dict, device) -> dict:
    infoDict['observation'] = infoDict['observation'].div(255).detach_()
    infoDict['next_observation'] = infoDict['next_observation'].div(255).detach_()
    return infoDict

def composeLHistv1(frame: TENSOR, position: Tuple[int,int], time: float):
    frame = frame.flatten(1)
    extra = torch.tensor([*position, time], dtype = F_TDTYPE_DEFT, device = frame.device).unsqueeze_(1)
    return Tcat([frame, extra], dim = 1).detach_()

def decomposeLHistv1(observation, frameShape) -> Tuple[TENSOR, TENSOR]:
    """
        returns
        -------
        frames: Tensor
        pos: Tensor
    """
    position = observation[:,-3:-1]
    time = observation[:,-1]
    frames = observation[:,:-3].reshape(-1, frameShape)
    return frames, position
