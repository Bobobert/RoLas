from .base import Agent
from rofl.functions.const import *
from rofl.functions.functions import clipReward, nprnd, isTerminalAtari
from rofl.utils.memory import dqnMemory
from rofl.utils.dqn import genFrameStack, lHistObsProcess, reportQmean
from rofl.utils.openCV import imgResize, YChannelResize

class dqnAtariAgent(Agent):
    name = 'dqn agent v1'

    def initAgent(self, **kwargs):
        self.lives = None
        self.lhist = self.config['agent']['lhist']
        self.memory = dqnMemory(self.config)
        self.clipReward = self.config['agent'].get('clip_reward', 0)
        self.frameSize = tuple(self.config['env']['obs_shape'])
        self.frameStack = genFrameStack(self.config)
        self.isAtari = self.config['env'].get('atari', True)
        self.envActions = self.config['policy']['n_actions']
        self.fixedTrajectory = None

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
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
        
    def envStep(self, action):
        obsDict = super().envStep(action)
        # sligth mod to the obsDict from DQN memory
        obsDict['next_frame'] = self.frameStack.copy()
        obsDict['observation'] = None # Deletes the reference to the tensor generated, mem expensive while GPU
        obsDict['device'] = DEVICE_DEFT
        return obsDict
    
    def reportCustomMetric(self):
        return reportQmean(self)

    def rndAction(self):
        return nprnd.randint(self.envActions)
