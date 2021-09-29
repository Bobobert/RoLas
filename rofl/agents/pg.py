from .base import Agent
from rofl.functions.const import DEVICE_DEFT, I_TDTYPE_DEFT
from rofl.functions.functions import clipReward, np, torch, rnd, no_grad, ceil
from rofl.functions.torch import array2Tensor
from rofl.functions.coach import singlePathRollout, episodicRollout
from rofl.utils.openCV import imgResize

class pgAgent(Agent):
    name = "pg gym agent"
    def initAgent(self, **kwargs):
        config = self.config
        self.clipReward = config["agent"].get("clip_reward", 0.0)

    def processReward(self, reward):
        return clipReward(self, reward)

    def processObs(self, obs, reset = False):
        return array2Tensor(obs, device = self.device)

    def getBatch(self, size: int, proportion: float = 1, random=False, 
                    device=DEVICE_DEFT, progBar: bool = False):
        return super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)

    def getEpisode(self, random=False):
        keys = []
        keys.append(('action', I_TDTYPE_DEFT)) if self.policy.discrete else None
        episode = episodicRollout(self, *keys, random = random, device = self.device)
        episode['observation'] = episode['observation'].squeeze().requires_grad_(True)
        return episode

    # the idea of having in coach the singlePathRollout is to use it as to do n-step updates!
    # no new memory should be required
    # TODO, generate a batch of those type of rollouts

    #def getEpisode(self, random):
    #    return singlePathRollout(self, self.maxEpLen, random = random)
    

class pgFFAgent(pgAgent):
    name = "forestFire_pgAgent"
    def __init__(self, config, policy, envMaker, tbw = None):

        super(pgFFAgent, self).__init__(config, policy, envMaker, tbw)
        self.isAtari = config["env"]["atari"]
        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        #self.memory = MemoryFF(config)
        self.obsShape = (lhist, *obsShape)
        self.frameSize = obsShape
        self.frameStack, self.lastObs, self.lastFrame = np.zeros(self.obsShape, dtype = np.uint8), None, None
        
    def processObs(self, obs, reset: bool = False): # TODO: pass this to a function that uses lHistObsProcess
        # with reward type, compose the outputs as a tensor alone always.
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        frame, pos, tm = obs["frame"], obs["position"], obs.get("time", 0)
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        self.lastFrame = imgResize(frame, size = self.frameSize)
        self.frameStack[0] = self.lastFrame
        self.lastFrame = {"frame":self.lastFrame, "position":pos, "time":tm}
        newObs = torch.from_numpy(self.frameStack).to(self.device).unsqueeze(0).float().div(255)
        Tpos = torch.as_tensor(pos).to(self.device).float().unsqueeze(0)
        Ttm = torch.as_tensor([tm]).to(self.device).float().unsqueeze(0)
        return {"frame": newObs, "position":Tpos, "time":Ttm}