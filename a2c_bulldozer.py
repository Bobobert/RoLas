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
    }

agentConfig = {
    'agentClass' : 'agentMultiEnv',
    'workerClass' : 'caPgAgent',
    'lhist' : 4,
    'channels' : 4,
    'memory_size' : 10**2,
    'nstep' : 20,
    }

policyConfig = {
    'entropy_bonus' : 5e-1,
    'network' : {
        'networkClass' : 'ffActorCritic',
        'conv2d_1' : (32, 6, 4),
        'conv2d_2' : (64, 4, 2),
        'conv2d_3' : (64, 3, 1),
        'linear_1' : 512,
        'learning_rate' : 5e-5,
        },
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
    config, agent, policy, train, manager = setUpExperiment('a2c', expConfig, dummyManager = True, cuda = True)
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
