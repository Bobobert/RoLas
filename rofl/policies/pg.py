from torch.autograd.grad_mode import no_grad
from .base import Policy
from rofl.networks.base import ActorCritic
from rofl.functions.functions import Tmean, Tmul, Tsum, F, torch, reduceBatch, no_grad
from rofl.functions.const import DEVICE_DEFT, ENTROPY_LOSS
from rofl.functions.torch import clipGrads, getOptimizer
from rofl.utils.policies import genMiniBatchLin, getActionWProb, getActionWValProb, getBaselines, getParamsBaseline
from rofl.config.config import createNetwork

class pgPolicy(Policy):
    """
        Vanilla REINFORCE policy gradient policy.
        Can handle and actor and a baseline.
        Update method expects results from an episodic
        task, therefore gae is not used.

        This policy creates a baseline network if given in the config dict
        config['policy']['baseline'], there's no need to create it before
        as the actor.
    """
    name = "policy gradient v0"

    def initPolicy(self, **kwargs):
        config = self.config
        self.keysForUpdate = ['observation', 'action', 'return']

        self.actorHasCritic, self.valueBased = False, False

        self.entropyBonus = -1.0 * abs(config['policy'].get('entropy_bonus', ENTROPY_LOSS))
        self.lossPolicyC = -1.0 * abs(config['policy']['loss_policy_const'])
        self.lossValueC = abs(config['policy']['loss_value_const'])
        self.minibatchSize = config['policy']['minibatch_size']
        self.newEpoch = True

        self.optimizer = getOptimizer(config, self.actor)

        if isinstance(self.actor, ActorCritic):
            self.actorHasCritic = True
            self.baseline = self.actor
            self.valueBased = True
            return

        if (baseline := config['policy']['baseline']['networkClass']) is not None:
            baseline = createNetwork(config, key = 'baseline').to(kwargs.get('device', DEVICE_DEFT))
            self.optimizerBl = getOptimizer(config, baseline, key = 'baseline')
            self.valueBased = True
        self.baseline = baseline

    def getValue(self, observation, action):
        if self.actorHasCritic:
            return super().getValue(observation, action)
        elif self.valueBased:
            return self.baseline.getValue(observation, action)
        else:
            return 0.0 # bite me...

    def update(self, batchDict):
        N, miniSize = batchDict['N'], self.minibatchSize
        observations, actions, returns = batchDict['observation'], batchDict['action'], batchDict['return']

        if N < miniSize:
            self.batchUpdate(observations, actions, returns)
        else:
            gen = genMiniBatchLin(miniSize, N, observations, actions, returns)
            for obsMini, actMini, rtrnMini in gen:
                self.batchUpdate(obsMini, actMini, rtrnMini)
        
        del batchDict
        self.epoch += 1
        self.newEpoch = True

    def batchUpdate(self, observations, actions, returns):
        params, baselines = getParamsBaseline(self, observations)

        log_probs, entropy = self.actor.processDist(params, actions)
        log_probs = reduceBatch(log_probs)

        _F, _FBl = Tmean, F.mse_loss
        advantages = returns - baselines.detach()
        lossPolicy = Tmul(log_probs, advantages.squeeze())
        lossPolicy = _F(lossPolicy)
        lossEntropy = _F(entropy)
        lossBaseline = _FBl(baselines, returns) if self.actorHasCritic else torch.zeros((), device=self.device)

        loss = self.lossPolicyC * lossPolicy + self.entropyBonus * lossEntropy + self.lossValueC * lossBaseline
        self.optimizer.zero_grad()
        loss.backward()
        if self.clipGrad > 0:
            clipGrads(self.actor, self.clipGrad)
        self.optimizer.step()

        if self.doBaseline:
            lossBaseline = _FBl(baselines, returns)
            self.optimizerBl.zero_grad()
            if self.clipGrad > 0:
                clipGrads(self.baseline, self.clipGrad)
            lossBaseline.backward()
            self.optimizerBl.step()

        tbw = self.tbw
        if tbw != None and (self.epoch % self.tbwFreq == 0) and self.newEpoch:
            tbw.add_scalar('train/Actor loss', -1 * lossPolicy.item(), self.epoch)
            tbw.add_scalar('train/Baseline loss', lossBaseline.item(), self.epoch)
            tbw.add_scalar('train/Total loss', loss.item(), self.epoch)
            self._evalTBWActor_()
        self.newEpoch = False

    @property
    def doBaseline(self):
        return self.baseline is not None and not self.actorHasCritic

    def getAVP(self, observation):
        if self.actorHasCritic:
            with no_grad():
                return getActionWValProb(self.actor, observation)
        with no_grad():
            action, logProb = getActionWProb(self.actor, observation)
            value = getBaselines(self, observation)

        return action, value, logProb
