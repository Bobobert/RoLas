from torch._C import Value
from torch.distributions.kl import register_kl, kl_divergence
from torch.distributions import Categorical, Normal
from torch.distributions.utils import probs_to_logits, logits_to_probs

from rofl.functions.functions import Tcat, Texp, Tlog, Tdiv, Tmean, Tmul,\
    Tpow, Tsum, torch

klDivergence = kl_divergence

@register_kl(Normal, Normal)
def kl_normals(dist1, dist2):
    mean1, mean2 = dist1.mean, dist2.mean
    logstd1, logstd2 = Tlog(dist1.stddev), Tlog(dist2.stddev)
    var1, var2 = dist1.variance, dist2.variance

    return Tmean(logstd2 - logstd1  + (var1 + Tpow(mean1 - mean2, 2)) / (2 * var2) - 0.5)

@register_kl(Categorical, Categorical)
def kl_cats(dist1, dist2):
    ax1 = Tdiv(dist1.logits, dist2.logits)
    return Tmean(Tmul(dist1.probs, ax1))

class MultiCategorical:
    def __init__(self, logits=None):
        if logits is None:
            raise ValueError('Must provice at least one logit, None was passed')
            
        if len(logits) == 1:
            print('Creating a MultiCategorical with just one logits tensor.\
                    Consider using Catergorical instead.')

        distributions, erste, batchDim = [], True, None
        for logit in logits:
            if erste:
                erste = False
                batchDim = logit.shape[0]
            else:
                assert batchDim == logit.shape[0],\
                    'Batch sizes mismatch, had %d but %d was given' % (batchDim, logit.shape[0])
            distributions.append(Categorical(logits=logit))
        self.distributions = distributions

    @property
    def logits(self):
        return [dist.logits for dist in self.distributions]

    def sample(self):
        res =  [dist.sample().unsqueeze_(-1) for dist in self.distributions]
        
        return Tcat(res, dim=1)

    def log_prob(self, values):
        logProbs = []
        for dim, dist in enumerate(self.distributions):
            logProbs.append(dist.log_prob(values[:,dim]).unsqueeze_(-1))
        logProbs = Tcat(logProbs, dim=-1)
        logProbs = Tsum(logProbs, dim=1)
        return logProbs

    def entropy(self):
        erste, running = True, None
        for dist_ in self.distributions:
            probs = dist_.probs
            if erste:
                running = probs
                erste = False
            else:
                newRunning = torch.zeros((running.shape[0], running.shape[1] * probs.shape[1]),
                                         device=running.device, requires_grad=True)
                n = running.shape[1]

                for i in range(probs.shape[1]):
                    res = Tmul(running, probs[:,i].unsqueeze(-1))
                    newRunning[:,i * n:(i + 1) * n] = res
                running = newRunning
        prod = Tmul(logits_to_probs(running), running)
        return -Tsum(prod)
