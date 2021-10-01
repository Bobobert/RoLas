from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'atariEnvMaker',
    'name': 'breakout',
    'atari': True,
    'obs_shape': (84,84),
    'frameskip': 4,
    'max_length': -1,
    'warmup' : 'noop',
    }
agentConfig = {
    'agentClass' : 'dqnAtariAgent',
    'memory_size' : 7 * 10**5, # around 22GB of ram using CUDA
    }
policyConfig = {
    'policyClass' : 'dqnPolicy',
    'n_actions' : 4,
    'epsilon_life' : 3 * 10**5,
    'double' : True,
    'network' : {
        'networkClass' : 'dqnAtari',
        'net_hidden_1' : 512,
        'learning_rate' : 5e-5,
    },
}
trainConfig = {
    'epochs' : 2 * 10**6,
    'fill_memory' : 10**5,
    'test_freq' : 5 * 10**4,
    'max_steps_per_test' : 10**4,
    'expected_perfomance' : 100,
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
