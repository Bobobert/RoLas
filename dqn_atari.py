from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'atariEnvMaker',
    'name': 'seaquest',
    'atari': True,
    'obs_shape': (84,84),
    'frameskip': 4,
    'max_length': -1,
    'warmup' : 'noop',
    }

agentConfig = {
    'agentClass' : 'dqnAtariAgent',
    'lhist' : 4,
    'memory_size' : 10**6,
    }

policyConfig = {
    'n_actions' : 6,
    'epsilon_life' : 25 * 10**4,
    'double' : True,
    'network' : {
        'networkClass' : 'dqnAtari',
        'conv2d_1' : (32, 8, 4),
        'conv2d_2' : (64, 4, 2),
        'conv2d_3' : (64, 3, 1),
        'linear_1' : 512,
        'learning_rate' : 5e-5,
    },
}

trainConfig = {
    'epochs' : 10**6,
    'fill_memory' : 10**5,
    'test_freq' : 5 * 10**4,
    'max_steps_per_test' : 10**4,
    'max_time' : 100,
}

expConfig = {
    'agent' : agentConfig,
    'policy' : policyConfig,
    'train' : trainConfig,
    'env' : envConfig,
}

if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('dqn', expConfig, dummyManager = True)
    results = train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
