# The idea is to create from a MasterAgent a succesful platform to create many agent Workers which can generate
# batches of experience, from which the gradients can be calculated
# TODO, compare times from calculating the grad in the worker or to calculate the grads outside it.
# memory cost of grads vs memory cost of batches and grad per thread vs many thread for a grad 
# (usually the way torch works). 
# TODO, use shared memory in RAY? for CPU only, perhaps the weights could be cheaper...

from rofl.agents.pg import pgAgent
from rofl.functions.torch import getGradients, getListState, updateNet

class a2cAgent(pgAgent):
    name = 'A2C worker'
    
    def updatePolicy(self, piParams, blParams):
        self.policy.updateParams(piParams, blParams)

    def calculateGrad(self, random: bool = False):
        pi = self.policy
        batchDict = self.getEpisode(random = random)
        observation, action, returns = batchDict['observation'], batchDict['action'], batchDict['return']
        
        pi.batchUpdate(observation, action, returns)
        piGrad = getGradients(pi.actor)
        blGrad = getGradients(pi.baseline) if pi.baseline is not None else []

        return piGrad, blGrad

    def updateParams(self, actorParams, blParams = []):
        pi = self.policy
        updateNet(pi.actor, actorParams)
        if pi.baseline is not None and blParams != []:
            updateNet(pi.baseline, blParams)

    def getParams(self):
        pi = self.policy
        piParams = getListState(pi.actor)
        blParams = [] if pi.baseline is None else getListState(pi.baseline)

        return piParams, blParams
