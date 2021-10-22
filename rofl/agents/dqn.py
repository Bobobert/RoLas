from .base import Agent
from rofl.functions.const import *
from rofl.functions.functions import clipReward, nprnd, isTerminalAtari
from rofl.utils.memory import dqnMemory
from rofl.utils.dqn import dqnStepv0, dqnStepv1, lHistObsProcess, processBatchv0, processBatchv1, reportQmean
from rofl.utils.bulldozer import assertChannels, grid2ImgFollow, composeObsWContextv0
from rofl.utils.openCV import imgResize, YChannelResize

memKeys = [('action', I_TDTYPE_DEFT), ('observation', F_TDTYPE_DEFT), ('next_observation', F_TDTYPE_DEFT)]

class dqnAtariAgent(Agent):
    name = 'dqn agent v1'

    def initAgent(self, **kwargs):
        config = self.config
        self.lives = None
        self.clipReward = abs(config['agent'].get('clip_reward', 0))

        self.lhist, self.channels = config['agent']['lhist'], config['agent'].get('channels', 1)
        self.memory = dqnMemory(config, *memKeys)

        self.frameSize = tuple(config['env']['obs_shape'])
        self.lastFrame, self.prevFrame, self.zeroFrame = None, None, None

        self.isAtari = config['env'].get('atari', False)
        self.envActions = config['policy']['n_actions']
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

    def getBatch(self, size: int, proportion: float = 1, random=False, device=DEVICE_DEFT, progBar: bool = False):
        batch = super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)
        processBatchv0(batch)
        return batch

class dqnCaAgent(Agent):
    name = 'dqn CA v1'

    def initAgent(self, **kwargs):
        config = self.config

        assertChannels(config)
        self.lhist, self.channels = config['agent']['lhist'], config['agent'].get('channels', 1)
        self.useChannels, self.displayAgent = False, True
        if self.channels == 4:
            self.useChannels = True
        self.useContext = config['agent'].get('use_context', True)

        import rofl.envs.rewards as rewards
        rewardFunTarget = config['env']['reward_function']
        if rewardFunTarget is not None:
            rewardFunTarget = getattr(rewards, rewardFunTarget, None)
        self.rewardFunc = rewardFunTarget

        self.memory = dqnMemory(config, *memKeys)
        self.clipReward = abs(config['agent'].get('clip_reward', 0))
        self.frameSize = tuple(self.config['env']['obs_shape'])

        self.lastFrame, self.prevFrame, self.zeroFrame = None, None, None
        self.lastContext, self.prevContext, self.zeroContext = None, None, None
        self.envActions = self.config['policy']['n_actions']
        self.fixedTrajectory = None

    def processObs(self, obs, reset: bool = False):
        self.prevFrame = self.lastFrame if not reset else self.zeroFrame
        self.prevContext = self.lastContext if not reset else self.zeroContext
        
        grid, context = obs
        self.lastContext = context
        self.lastFrame = img = grid2ImgFollow(self.env, grid, context, self.frameSize, self.useChannels, self.displayAgent)

        tensorStack = lHistObsProcess(self, img, reset)

        if not self.useContext:
            context = self.zeroContext
        return composeObsWContextv0(tensorStack, context)

    def processReward(self, reward, **kwargs):
        rewardFunc = self.rewardFunc

        if rewardFunc is None:
            pass
        else:
            reward = reward(self, reward, **kwargs)

        return clipReward(self, reward)

    @dqnStepv1
    def envStep(self, action, **kwargs):
        return super().envStep(action, **kwargs)
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def getBatch(self, size: int, proportion: float = 1, random=False, device=DEVICE_DEFT, progBar: bool = False):
        batch = super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)
        processBatchv1(batch, self.useChannels, self.actionSpace)
        return batch
