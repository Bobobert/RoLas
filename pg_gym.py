from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'LunarLanderContinuous-v2',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'agentClass' : 'pgAgent',
    'memory_size' : 10**3,
    'gamma' : 0.99,
    'nstep' : 15,
    }

policyConfig = {
    'policyClass' : 'pgPolicy',
    'continuos' : True,
    'entropy_bonus' : 5e-3,
    'network' : {
        'networkClass' : 'gymAC',
        'linear_1' : 56,
        'linear_2' : 32,
        'learning_rate' : 1e-5,
    },
    'baseline' :{
        'networkClass' : 'gymBaseline',
        'linear_1' : 56,
        'learning_rate': 1e-4,
    }
}

trainConfig = {
    'epochs' : 10**6,
    'test_freq' : 5 * 10**3,
    'expected_performance': 100,
    'max_time' : 25,
}

expConfig = {
    'agent' : agentConfig,
    'policy' : policyConfig,
    'train' : trainConfig,
    'env' : envConfig,
}

if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('pg', expConfig, dummyManager = True, cuda = False)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
