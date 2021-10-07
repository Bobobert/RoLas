from .base import Policy
from rofl.networks.base import ActorCritic
from rofl.functions.functions import Tmean, Tmul, Tsum, F, torch, reduceBatch, np
from rofl.functions.const import DEVICE_DEFT, ENTROPY_LOSS
from rofl.functions.torch import clipGrads, getOptimizer
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
        self.actorHasCritic, self.valueBased = False, False

        self.entropyBonus = abs(config['policy'].get('entropy_bonus', ENTROPY_LOSS))
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

    def getAction(self, state):
        return self.actor.getAction(state)

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
            self.batchUpdate(observations, actions, returns) # TODO, a minibatch generator to slip large batches
        else:
            for lower in range(0, N, miniSize):
                obsMini = observations[lower:lower + miniSize]
                actMini = actions[lower:lower + miniSize]
                rtrnMini = returns[lower:lower + miniSize]
                self.batchUpdate(obsMini, actMini, rtrnMini)
        
        del batchDict
        self.epoch += 1
        self.newEpoch = True

    def batchUpdate(self, observations, actions, returns):
        if self.baseline is None:
            baselines = returns.new_zeros(returns.shape)
        elif not self.actorHasCritic:
            baselines = self.baseline(observations)
        else:
            baselines, params = self.actor(observations)
        
        params = params if self.actorHasCritic else self.actor.onlyActor(observations)
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
            lossB = _FBl(baselines, returns)
            self.optimizerBl.zero_grad()
            if self.clipGrad > 0:
                clipGrads(self.baseline, self.clipGrad)
            lossB.backward()
            self.optimizerBl.step()

        if self.tbw != None and (self.epoch % self.tbwFreq == 0) and self.newEpoch:
            self.tbw.add_scalar('train/Actor loss', -1 * lossPolicy.item(), self.epoch)
            self.tbw.add_scalar('train/Total loss', loss.item(), self.epoch)
            self._evalTBWActor_()
        self.newEpoch = False

    @property
    def doBaseline(self):
        return self.baseline is not None and not self.actorHasCritic

def catchChanges(oldParams, newParams, th = 10**3):
    maxChange = -np.inf
    for op, p in zip(oldParams, newParams):
        diff = p.detach() - op.detach()
        mD = diff.max().to(DEVICE_DEFT).item() 
        if mD > maxChange:
            maxChange = mD
        if maxChange > th:
            raise ValueError('This is BAD news')
        if torch.isnan(p).any():
            raise ValueError('THIS IS REALLY BAD NEWS')
