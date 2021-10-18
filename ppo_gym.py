from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'LunarLander-v2',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'gamma' : 0.99,
    'nstep' : 30,
    'workers' : 16,
    'memory_size' : 10**2,
    'agentClass' : 'agentMultiEnv'
    }

policyConfig = {
    'continuos' : False,
    'entropy_bonus' : 5e-1,
    'network' : {
        'networkClass' : 'gymAC',
        'linear_1' : 32,
        'learning_rate' : 1e-4,
    },
    'epochs' : 10,
}

trainConfig = {
    'epochs' : 10**4,
    'test_freq' : 10**2,
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
    config, agent, policy, train, manager = setUpExperiment('ppo', expConfig, dummyManager = False, cuda = False)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
