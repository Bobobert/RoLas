from .base import BaseAgent
from rofl.functions.const import F_TDTYPE_DEFT, DEVICE_DEFT
from rofl.functions.functions import clipReward, nprnd, isTerminalAtari
from rofl.utils.memory import DqnMemory
from rofl.utils.dqn import dqnStepv0, dqnStepv1, lHistObsProcess,\
    processBatchv0, reportQmean, reportRatio, extraKeysForBatches
from rofl.utils.bulldozer import assertChannels, calRatio,\
    grid2ImgFollow, composeObsWContextv0, prepare4Ratio, processBatchv1
from rofl.utils.openCV import imgResize, YChannelResize

memKeys = [('observation', F_TDTYPE_DEFT), ('next_observation', F_TDTYPE_DEFT)]

class DqnAtariAgent(BaseAgent):
    name = 'dqn agent v1'

    def initAgent(self, **kwargs):
        config = self.config
        self.lives = None
        self.clipReward = abs(config['agent'].get('clip_reward', 0))

        self.lhist, self.channels = config['agent']['lhist'], config['agent'].get('channels', 1)
        self.memory = DqnMemory(config, *memKeys)

        self.frameSize = tuple(config['env']['obs_shape'])
        self.lastFrame, self.prevFrame, self.zeroFrame = None, None, None

        self.isAtari = config['env'].get('atari', False)
        self.envActions = config['policy']['n_actions']
        self.fixedTrajectory = None

    def processObs(self, obs, info, done, reset: bool = False):
        self.prevFrame = self.lastFrame if not reset else self.zeroFrame
        if self.isAtari:
            obs = self.lastFrame = YChannelResize(obs, size=self.frameSize)
        else:
            obs = self.lastFrame = imgResize(obs, size=self.frameSize)
        return lHistObsProcess(self, obs, reset)

    def isTerminal(self, obs, reward, info, done):
        if super().isTerminal(obs, reward, info, done):
            return True
        return isTerminalAtari(self, info)
            
    def processReward(self, obs, reward, info, done):
        return clipReward(self, reward)
    
    def envStep(self, action, **kwargs):
        obsDict = super().envStep(action, **kwargs)
        return dqnStepv0(self, obsDict)
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def rndAction(self):
        return nprnd.randint(self.envActions)

    def getBatch(self, size: int, proportion: float=1, random=False, device=DEVICE_DEFT, progBar: bool=False):
        batch = super().getBatch(size, proportion=proportion, random=random, device=device, progBar=progBar)
        processBatchv0(batch)
        return batch

class DqnCAAgent(BaseAgent):
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

        self.memory = DqnMemory(config, *memKeys)
        self.clipReward = abs(config['agent'].get('clip_reward', 0))
        self.frameSize = tuple(self.config['env']['obs_shape'])

        self.lastFrame, self.prevFrame, self.zeroFrame = None, None, None
        self.lastContext, self.prevContext, self.zeroContext = None, None, None
        self.envActions = self.config['policy']['n_actions']
        self.fixedTrajectory = None
        if self.keysForBatches is not None:
            self.keysForBatches = self.keysForBatches.copy() + extraKeysForBatches

    def processObs(self, obs, info, done, reset: bool = False):
        self.prevFrame = self.lastFrame if not reset else self.zeroFrame
        self.prevContext = self.lastContext if not reset else self.zeroContext
        
        grid, context = obs
        self.lastContext = context
        self.lastFrame = img = grid2ImgFollow(self.env, grid, context, 
                                                self.frameSize, self.useChannels, 
                                                self.displayAgent)

        tensorStack = lHistObsProcess(self, img, reset)

        if not self.useContext:
            context = self.zeroContext
        return composeObsWContextv0(tensorStack, context)

    def processReward(self, obs, reward, info, done):
        rewardFunc = self.rewardFunc

        if rewardFunc is None:
            pass
        else:
            reward = rewardFunc(self, obs, reward, info, done)

        return clipReward(self, reward)

    def envStep(self, action, **kwargs):
        obsDict =  super().envStep(action, **kwargs)
        return dqnStepv1(self, obsDict)
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def getBatch(self, size: int, proportion: float = 1, 
                    random=False, device=DEVICE_DEFT, 
                    progBar: bool = False):
        batch = super().getBatch(size, proportion=proportion,
                                    random=random, device=device,
                                    progBar=progBar)
        processBatchv1(batch, self.useChannels, self.actionSpace)
        return batch

    def prepareTest(self):
        prepare4Ratio(self)

    def calculateCustomMetric(self, env, reward, done):
        calRatio(self, env)

    def reportCustomMetric(self):
        return reportRatio(self)
