from rofl.algorithms.pg import train
from rofl.algorithms.pg import algConfig as cnf
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
        'cg_iterations' : 10,
        'ls_iterations' : 10,
        'policyClass' : 'trpoPolicy',
        'epochs' : 20,
        'network' : {
            'optimizer' : 'dummy',
        },
    },
    'train' : {
        'modeGrad' : False,
    }
}

algConfig = completeDict(algConfig, cnf)
