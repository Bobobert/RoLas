from rofl.functions.torch import accumulateGrad, getParams
from rofl.utils.policies import setEmptyOpt
from rofl.policies.pg import pgPolicy

class a2cPolicy(pgPolicy):
    """
        Main policy for a2C not meant for the workers
    """
    name = 'a2c v0'

    def gradUpdate(self, *grads):
        #piGrad, blGrad = [], []
        #for grad in grads:
        #    piGrad.append(grad[0])
        #    blGrad.append(grad[1])

        for piGrad, _ in grads:
            self.optimizer.zero_grad()
            accumulateGrad(self.actor, piGrad)
            self.optimizer.step()

        if self.doBaseline:
            for _, blGrad in grads:
                self.optimizerBl.zero_grad()
                accumulateGrad(self.baseline, blGrad)
                self.optimizerBl.step()

        if self.tbw != None and (self.epoch % self.tbwFreq == 0):
            self._evalTBWActor_()

        self.newEpoch = True
        self.epoch += 1

    def update(self, *batchDict):
        for dict_ in batchDict:
            obs, act, rtrn = dict_['observation'], dict_['action'], dict_['return']
            self.batchUpdate(obs, act, rtrn)

        self.newEpoch = True
        self.epoch += 1
    
    def getParams(self):
        return getParams(self)

class a2cWorkerPolicy(pgPolicy):
    name = 'a2c v0 - worker'

    def initPolicy(self, **kwargs):
        setEmptyOpt(self)
        super().initPolicy(**kwargs)
