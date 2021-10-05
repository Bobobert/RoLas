from rofl import setUpExperiment
from rofl.functions.const import MINIBATCH_SIZE

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'LunarLanderContinuous-v2',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'gamma' : 0.99,
    'nstep' : 20,
    'workers' : 8,
    }

policyConfig = {
    'continuos' : True,
    'minibatch_size' : MINIBATCH_SIZE,
    'entropy_bonus' : 5e-3,
    'network' : {
        'networkClass' : 'gymAC',
        'linear_1' : 56,
        'linear_2' : 32,
        'learning_rate' : 1e-5,
    },
    'baseline' :{
        'networkClass' : None,#'gymBaseline',
        'linear_1' : 56,
        'linear_2' : 32,
        'learning_rate': 1e-5,
    }
}

trainConfig = {
    'epochs' : 10**6,
    'test_freq' : 5 * 10**3,
    'expected_performance': 100,
    'max_time' : 30,
}

expConfig = {
    'agent' : agentConfig,
    'policy' : policyConfig,
    'train' : trainConfig,
    'env' : envConfig,
}

if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('a2c', expConfig, dummyManager = True, cuda = False)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
