from rofl.functions.functions import torch, Texp, reduceBatch, Tmean, Tmul, F
from rofl.functions.torch import clipGrads
from rofl.policies.a2c import a2cPolicy
from rofl.utils.policies import getParamsBaseline, setEmptyOpt, calculateGAE, trainBaseline

def putVariables(policy):
    config = policy.config

    policy.epsSurrogate = config['policy']['epsilon_surrogate']
    policy.epochsPerBatch = config['policy']['epochs']
    policy.maxKLDiff = config['policy']['max_diff_kl']
    policy.keysForUpdate = ['observation', 'return', 'action', 'log_prob']
    if policy.gae:
        policy.keysForUpdate += ['reward', 'next_observation', 'done']

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
        observations, actions, returns = batchDict['observation'], batchDict['action'], batchDict['return']
        log_probs_old = reduceBatch(batchDict['log_prob'])
        
        params, baselines = getParamsBaseline(self, observations)
        if self.gae:
            advantages = calculateGAE(self, baselines, batchDict['next_observation'],\
                batchDict['done'], batchDict['reward'], self.gamma, self.lmbd)
        else:
            advantages = returns - baselines.detach()
        advantages.squeeze_()
        
        eps = self.epsSurrogate
        _F, _FBl = Tmean, F.mse_loss
        brokeKL = 0
        
        for i in range(self.epochsPerBatch):
            
            if i > 0:
                params, baselines = getParamsBaseline(self, observations)
            log_probs, entropy = self.actor.processDist(params, actions)
            log_probs = reduceBatch(log_probs)

            ratio = Texp(log_probs - log_probs_old)            
            clipped = Tmul(ratio.clamp(min = 1.0 - eps, max = 1.0 + eps), advantages)
            unclipped = Tmul(ratio, advantages)
            lossPolicy = torch.fmin(unclipped, clipped)

            lossPolicy = _F(lossPolicy) # average falls flat first iters??
            lossEntropy = _F(entropy)
            lossBaseline = _FBl(baselines, returns) if self.actorHasCritic else torch.zeros((), device=self.device)
            loss = self.lossPolicyC * lossPolicy + self.entropyBonus * lossEntropy + self.lossValueC * lossBaseline

            # check KL difference through the log_probs
            kl = Tmean(log_probs_old.detach() - log_probs.detach()).cpu().item()
            if kl > self.maxKLDiff: 
                brokeKL = i
                break

            self.optimizer.zero_grad()
            loss.backward()
            if self.clipGrad > 0:
                clipGrads(self.actor, self.clipGrad)
            self.optimizer.step()

            if self.doBaseline:
                lossBaseline = trainBaseline(self, baselines, returns, _FBl)

        tbw = self.tbw
        if tbw != None and (self.epoch % self.tbwFreq == 0) and self.newEpoch:
            epoch = self.epoch
            tbw.add_scalar('train/Actor loss', -1 * lossPolicy.cpu().item(), epoch)
            tbw.add_scalar('train/Baseline loss', lossBaseline.cpu().item(), epoch)
            tbw.add_scalar('train/Entropy distribution', lossEntropy.cpu().item(), epoch)
            tbw.add_scalar('train/Total loss', loss.cpu().item(), epoch)
            tbw.add_scalar('train/KL early stopping', brokeKL, epoch)
            self._evalTBWActor_()
        self.newEpoch = False

class ppoWorkerPolicy(ppoPolicy):
    name = 'ppo v0 - worker'

    def initPolicy(self, **kwargs):
        setEmptyOpt(self)
        super().initPolicy(**kwargs)
