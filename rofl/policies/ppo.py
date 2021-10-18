from rofl.functions import dicts
from rofl.functions.functions import torch, Texp, reduceBatch, Tmean, Tmul, F, no_grad
from rofl.functions.torch import clipGrads, normMean
from rofl.policies.a2c import a2cPolicy
from rofl.utils.policies import getParamsBaseline, setEmptyOpt, calculateGAE

def putVariables(policy):
    config = policy.config
    policy.epsSurrogate = config['policy']['epsilon_surrogate']
    policy.epochsPerBatch = config['policy']['epochs']
    policy.maxKLDiff = config['policy']['max_diff_kl']
    policy.keysForUpdate = ['observation', 'next_observation', 'return', 'reward', 'action', 'done', 'log_prob']

class ppoPolicy(a2cPolicy):
    name = 'ppo v0'

    def initPolicy(self, **kwargs):
        super().initPolicy(**kwargs)
        putVariables(self)

    def update(self, *batchDict):
        for dict_ in batchDict:
            self.batchUpdate(dict_)

        self.newEpoch = True
        self.epoch += 1

    def batchUpdate(self, batchDict):
        observations, nObservations = batchDict['observation'], batchDict['next_observation']
        actions, rewards, returns = batchDict['action'], batchDict['reward'], batchDict['return']
        dones = batchDict['done']

        log_probs_old = reduceBatch(batchDict['log_prob'])

        params, baselines = getParamsBaseline(self, observations)
        if self.gae:
            advantages = calculateGAE(self, baselines, nObservations, dones, rewards, self.gamma, self.lmbd)
        else:
            advantages = returns - baselines.detach()
        
        eps = self.epsSurrogate
        _F, _FBl = Tmean, F.mse_loss
        brokeKL = 0
        
        for i in range(self.epochsPerBatch):
            
            if i > 0:
                params, baselines = getParamsBaseline(self, observations)
            log_probs, entropy = self.actor.processDist(params, actions)
            log_probs = reduceBatch(log_probs)

            ratio = Texp(log_probs - log_probs_old)            
            clipped = Tmul(ratio.clamp(min = 1.0 - eps, max = 1.0 + eps), advantages.squeeze())
            unclipped = Tmul(ratio, advantages.squeeze())
            lossPolicy = torch.fmin(unclipped, clipped)

            lossPolicy = _F(lossPolicy) # average falls flat first iters??
            lossEntropy = _F(entropy)
            lossBaseline = _FBl(baselines, returns) if self.actorHasCritic else torch.zeros((), device=self.device)

            # check KL difference through the log_probs
            kl = Tmean(log_probs_old.detach() - log_probs.detach()).cpu().item()
            if kl > self.maxKLDiff: 
                brokeKL = i
                break

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
            epoch = self.epoch
            tbw.add_scalar('train/Actor loss', -1 * lossPolicy.item(), epoch)
            tbw.add_scalar('train/Baseline loss', lossBaseline.item(), epoch)
            tbw.add_scalar('train/Total loss', loss.item(), epoch)
            tbw.add_scalar('train/KL early stopping', brokeKL, epoch)
            self._evalTBWActor_()
        self.newEpoch = False

class ppoWorkerPolicy(ppoPolicy):
    name = 'ppo v0 - worker'

    def initPolicy(self, **kwargs):
        setEmptyOpt(self)
        super().initPolicy(**kwargs)
