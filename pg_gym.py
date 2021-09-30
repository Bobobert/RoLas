from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'CartPole-v1',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentConfig = {
    'agentClass' : 'pgAgent',
    'memory_size' : 10**3,
    'gamma' : 0.99,
    }

policyConfig = {
    'policyClass' : 'pgPolicy',
    'n_actions' : 2,
    'entropy_bonus' : 5e-3,
    'network' : {
        'networkClass' : 'gymActor',
        'linear_hidden_1' : 32,
        'learning_rate' : 5e-5,
    },
    'baseline' :{
        'networkClass' : 'gymBaseline',
        'linear_hidden_1' : 32,
        'learning_rate': 1e-4,
    }
}

trainConfig = {
    'epochs' : 10**5,
    'test_freq' : 10**3,
    'expected_performance': 150,
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
