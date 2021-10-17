# The idea is to create from a MasterAgent a succesful platform to create many agent Workers which can generate
# batches of experience, from which the gradients can be calculated
from rofl.agents.pg import pgAgent
from rofl.functions.torch import getNGradients, getParams, updateNet
from rofl.utils.random import seeder

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
    