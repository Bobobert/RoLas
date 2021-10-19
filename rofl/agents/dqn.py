from .base import Agent
from rofl.functions.const import *
from rofl.functions.functions import clipReward, newZero, nprnd, isTerminalAtari
from rofl.utils.memory import dqnMemory
from rofl.utils.dqn import dqnStepv0, genFrameStack, lHistObsProcess, processBatchv0, reportQmean
from rofl.utils.openCV import imgResize, YChannelResize

memKeys = [('action', I_TDTYPE_DEFT), ('observation', F_TDTYPE_DEFT), ('next_observation', F_TDTYPE_DEFT)]

class dqnAtariAgent(Agent):
    name = 'dqn agent v1'

    def initAgent(self, **kwargs):
        self.lives = None
        self.lhist = self.config['agent']['lhist']
        self.memory = dqnMemory(self.config, *memKeys)
        self.clipReward = abs(self.config['agent'].get('clip_reward', 0))
        self.frameSize = tuple(self.config['env']['obs_shape'])
        self.lastFrame, self.prevFrame, self.zeroFrame = None, None, None
        self.isAtari = self.config['env'].get('atari', False)
        self.envActions = self.config['policy']['n_actions']
        self.fixedTrajectory = None

    def processObs(self, obs, reset: bool = False):
        self.prevFrame = self.lastFrame if not reset else self.zeroFrame
        if self.isAtari:
            obs = self.lastFrame = YChannelResize(obs, size = self.frameSize)
        else:
            obs = self.lastFrame = imgResize(obs, size = self.frameSize)
        return lHistObsProcess(self, obs, reset)

    def isTerminal(self, obs, done, info, **kwargs):
        if super().isTerminal(obs, done, info):
            return True
        return isTerminalAtari(self, info)
            
    def processReward(self, reward, **kwargs):
        return clipReward(self, reward)
    
    @dqnStepv0
    def envStep(self, action, **kwargs):
        return super().envStep(action, **kwargs)
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def rndAction(self):
        return nprnd.randint(self.envActions)

    def getEpisode(self, random=False, device=None):
        episode = super().getEpisode(random=random, device=device)
        processBatchv0(episode)
        return episode

    def getBatch(self, size: int, proportion: float = 1, random=False, device=DEVICE_DEFT, progBar: bool = False):
        batch = super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)
        processBatchv0(batch)
        return batch

class dqnCAAgent(Agent):
    name = 'dqn CA v1'

    def initAgent(self, **kwargs):
        self.lhist = self.config['agent']['lhist']
        self.memory = dqnMemory(self.config)
        self.frameSize = tuple(self.config['env']['obs_shape'])
        self.frameStack = self.lastFrameStack = genFrameStack(self.config)
        self.envActions = self.config['policy']['n_actions']
        self.fixedTrajectory = None

    def processObs(self, obs, reset: bool = False):
        obs = self.prevFrame = imgResize(obs, size = self.frameSize)
        return lHistObsProcess(self, obs, reset)
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def rndAction(self):
        return nprnd.randint(self.envActions)

    def getEpisode(self, random=False, device=None):
        episode = super().getEpisode(random=random, device=device)
        processBatchv0(episode)
        return episode

    def getBatch(self, size: int, proportion: float = 1, random=False, device=DEVICE_DEFT, progBar: bool = False):
        batch = super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)
        processBatchv0(batch)
        return batch
