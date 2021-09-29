from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'CartPole-v0',
    'atari': False,
    'gamma': 0.997,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'agentClass' : 'pgAgent',
    'memory_size' : 10**4,
    }

policyConfig = {
    'policyClass' : 'pgPolicy',
    'n_actions' : 2,
    'network' : {
        'networkClass' : 'gymActor',
        'net_hidden_1' : 56,
        'learning_rate' : 5e-5,
    },
    'baseline' :{
        'networkClass' : 'gymBaseline',
        'net_hidden_1' : 56,
        'learning_rate': 1e-4,
    }
}

trainConfig = {
    'epochs' : 10**5,
    'test_freq' : 10**3,
    'expected_perfomance': 150,
}

expConfig = {
    'agent' : agentConfig,
    'policy' : policyConfig,
    'train' : trainConfig,
    'env' : envConfig,
}

if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('pg', expConfig, dummyManager = True)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
