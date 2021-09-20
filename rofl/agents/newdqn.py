from .base import Agent
from rofl.functions.const import *
from rofl.utils.memory import dqnMemory
from rofl.utils.dqn import genFrameStack, lHistObsProcess, reportQmean
from rofl.utils.cv import imgResize, YChannelResize

class dqnAtariAgent(Agent):
    name = 'dqn agent v1'

    def initAgent(self):
        self.lives = None
        self.lhist = self.config['agent']['lhist']
        self.memory = dqnMemory(self.config)
        self.clipReward = self.config['agent'].get('clip_reward', 0)
        self.frameSize = tuple(self.config['env']['obs_shape'])
        self.frameStack = genFrameStack(self.config)
        self.isAtari = self.config['env'].get('atari', True)

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
        done = False
        if self.isAtari:
            lives = info.get('ale.lives', 0)
            if self._reseted:
                self.lives = lives
            elif lives != self.lives:
                self.lives = lives
                done = True # marked as terminal but no reset required
        return done
            
    def processReward(self, reward, **kwargs):
        if self.clipReward != 0:
            return np.clip(reward, -self.clipReward, self.clipReward)
        return reward
        
    def envStep(self, action):
        obsDict = super().envStep(action)
        # sligth mod to the obsDict from DQN memory
        obsDict['frame'] = self.frameStack.copy()
        obsDict['observation'] = None # Deletes the reference to the tensor generated, mem expensive while GPU
        obsDict['device'] = DEVICE_DEFT
        return obsDict
    
    def reportCustomMetric(self):
        return reportQmean(self)