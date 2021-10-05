from rofl.functions.torch import accumulateGrad, getGradients, getListState, newNet, updateNet
from rofl.policies.pg import pgPolicy

class a2cPolicy(pgPolicy):
    """
        Main policy for a2C not meant for the workers
    """
    name = 'a2c v0'

    def gradUpdate(self, *grads):
        piGrad, blGrad = [], []
        for grad in grads:
            piGrad.append(grad[0])
            blGrad.append(grad[1])

        self.optimizer.zero_grad()
        accumulateGrad(self.actor, *piGrad)
        self.optimizer.step()

        if self.baseline is not None:
            self.optimizerBl.zero_grad()
            accumulateGrad(self.baseline, *blGrad)
            self.optimizerBl.step()

        if self.tbw != None and (self.epoch % self.tbwFreq == 0):
            self._evalTBWActor_()

        self.epoch += 1

    def update(self, *batchDict):
        for n, dict_ in enumerate(batchDict):
            obs, act, rtrn = dict_['observation'], dict_['action'], dict_['return']
            self.batchUpdate(obs, act, rtrn)

        self.newEpoch = True
        self.epoch += 1
    
    def getParams(self):
        pi = self
        piParams = getListState(pi.actor)
        blParams = [] if pi.baseline is None else getListState(pi.baseline)

        return piParams, blParams

class a2cWorkerPolicy(pgPolicy):
    name = 'a2c v0 - worker'

    def initPolicy(self, **kwargs):
        self.tbw = None
        config = self.config
        # set config for dummy optimizers
        config['policy']['network']['optimizer'] = 'dummy'
        if config['policy']['baseline']['networkClass'] is not None:
            config['policy']['baseline']['optimizer'] = 'dummy'
        super().initPolicy(**kwargs)
