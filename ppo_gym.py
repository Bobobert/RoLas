from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'CartPole-v1',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'gamma' : 0.99,
    'nstep' : 10,
    'workers' : 2,
    'memory_size' : 10**2,
    }

policyConfig = {
    'continuos' : False,
    'entropy_bonus' : 5e-2,
    'network' : {
        'networkClass' : 'gymAC',
        'linear_1' : 32,
        'learning_rate' : 5e-5,
    },
}

trainConfig = {
    'epochs' : 10**5,
    'test_freq' : 5 * 10**3,
    'expected_performance': 100,
    'max_time' : 10,
}

expConfig = {
    'agent' : agentConfig,
    'policy' : policyConfig,
    'train' : trainConfig,
    'env' : envConfig,
}

if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('ppo', expConfig, dummyManager = True, cuda = False)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
