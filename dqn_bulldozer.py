from rofl import setUpExperiment

envConfig = {
    'envMaker' : 'gymcaEnvMaker',
    'name': 'ForestFireBulldozer-v1',
    'n_rows' : 120,
    'n_cols' : 100,
    'wind_speed' : 20,
    'wind_direction' : 290,
    'obs_shape' : (36, 36),
    'max_length' : 500,
    'reward_function' : None,
    'warmup' : 'noop',
    'warmup_max_steps' : 15,
    }

agentConfig = {
    'agentClass' : 'dqnCaAgent',
    'lhist' : 4,
    'channels' : 4,
    'memory_size' : 10**6,
    }

policyConfig = {
    'epsilon_life' : 25 * 10**4,
    'double' : False,
    'network' : {
        'networkClass' : 'dqnCA',
        'conv2d_1' : (32, 6, 4),
        'conv2d_2' : (64, 4, 2),
        'conv2d_3' : (64, 3, 1),
        'linear_1' : 512,
        'learning_rate' : 5e-5,
    },
}

trainConfig = {
    'epochs' : 10**6,
    'fill_memory' : 10**2,
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
    config, agent, policy, train, manager = setUpExperiment('dqn', expConfig, dummyManager = True, cuda = True)
    results = train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
