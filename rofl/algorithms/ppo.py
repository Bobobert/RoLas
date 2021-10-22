from rofl.algorithms.a2c import train
from rofl.algorithms.a2c import algConfig as cnf
from rofl.config.config import completeDict
from rofl.functions.const import MAX_DKL, EPS_SURROGATE

algConfig = {
    'agent' : {
        'gae' : True,
        'need_log_prob' : True,
    },
    'policy': {
        'max_diff_kl' : MAX_DKL,
        'epsilon_surrogate' : EPS_SURROGATE,
        'policyClass' : 'ppoPolicy',
        'workerPolicyClass' : 'ppoWorkerPolicy',
        'epochs' : 20,
    },
    'train' : {
        'modeGrad' : False,
    }
}

algConfig = completeDict(algConfig, cnf)
