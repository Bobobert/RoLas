# The idea is to create from a MasterAgent a succesful platform to create many agent Workers which can generate
# batches of experience, from which the gradients can be calculated
from rofl.agents.pg import pgAgent
from rofl.agents.dqn import dqnCaAgent, memKeys
from rofl.functions.coach import singlePathRollout
from rofl.functions.torch import getNGradients, getParams, updateNet
from rofl.utils.dqn import processBatchv1
from rofl.utils.random import seeder
from rofl.utils.memory import episodicMemoryFrames

class a2cAgent(pgAgent):
    name = 'A2C worker'
    
    def initAgent(self, **kwargs):
        super().initAgent(**kwargs)
        config = self.config
        seeder(config['seed'] + config['agent']['id'], self.device)

    def calculateGrad(self, random: bool = False):
        pi = self.policy
        batchDict = self.getEpisode(random = random)
        observation, action, returns = batchDict['observation'], batchDict['action'], batchDict['return']
        
        pi.batchUpdate(observation, action, returns)
        piGrad = getNGradients(pi.actor)
        blGrad = getNGradients(pi.baseline) if pi.baseline is not None else []

        return piGrad, blGrad

    def updateParams(self, actorParams, blParams = []):
        pi = self.policy
        updateNet(pi.actor, actorParams)
        if pi.baseline is not None and blParams != []:
            updateNet(pi.baseline, blParams)

    def getParams(self):
        return getParams(self.policy)

class caPgAgent(dqnCaAgent):
    name = 'policy gradient CA worker'
    
    def initAgent(self, **kwargs):
        super().initAgent(**kwargs)
        config = self.config
        seeder(config['seed'] + config['agent']['id'], self.device)

        self.memory = episodicMemoryFrames(config, *memKeys) # TODO, dqn memory but episodic
        self.nstep = config['agent'].get('nstep', -1)
        self.forceLen = True if self.nstep > 0 else False

    def getEpisode(self, random = False, device = None):
        memory = self.memory
        memory.reset()

        singlePathRollout(self, self.nstep, memory, random = random, forceLen = self.forceLen)
        device = self.device if device is None else device
        episode = memory.getEpisode(device, None)#self.keysForBatches)

        return processBatchv1(episode, self.useChannels, self.actionSpace)

    def calculateGrad(self, random: bool = False):
        pi = self.policy
        pi.update(self.getEpisode(random = random))

        piGrad = getNGradients(pi.actor)
        blGrad = getNGradients(pi.baseline) if pi.baseline is not None else []

        return piGrad, blGrad

    def updateParams(self, actorParams, blParams = []):
        pi = self.policy
        updateNet(pi.actor, actorParams)
        if pi.baseline is not None and blParams != []:
            updateNet(pi.baseline, blParams)

    def getParams(self):
        return getParams(self.policy)

