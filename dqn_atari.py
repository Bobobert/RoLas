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
    'memory_size' : 7 * 10**5,
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

"""
from rofl.functions import getDevice, createConfig
from rofl.config import createPolicy, getEnvMaker, createAgent, getTrainFun
from rofl.utils import seeder, pathManager
device = getDevice()
seeder(8080, device)
def setUp():
    config = createConfig(envConfig, expName = 'dqn')

    config['agent']['agentClass'] = 'dqnAtariAgent'
    config['agent']['memory_size'] = 7*10**5
    config['policy']['policyClass'] = 'dqnPolicy'
    config['policy']['n_actions'] = 4
    config['policy']['network']['networkClass'] = 'dqnAtari'
    config['policy']['network']['net_hidden_1'] = 512
    config['policy']['network']['learning_rate'] = 5e-5
    config['policy']['epsilon_life'] = 3 * 10**5
    config['policy']['double'] = True
    config['policy']['freq_update_target'] = 10**4
    config['train']['epochs'] = 2*10**6
    config['train']['fill_memory'] = 10**5
    config['train']['test_freq'] = 5*10**4
    config['train']['max_steps_per_test'] = 10**4
    config['train']['test_iters'] = 20
    config['train']['expected_performance'] = 100

    manager = pathManager(config, dummy = True)
    writer = manager.startTBW()
    policy = createPolicy(config, tbw = writer, device = device)
    envMaker = getEnvMaker(config)
    agent = createAgent(config, policy, envMaker, tbw = writer)
    train = getTrainFun(config)
    manager.saveConfig()
    print(config)
    return config, agent, policy, manager, train
"""
if __name__ == '__main__':
    config, agent, policy, train, manager = setUpExperiment('dqn', expConfig, dummyManager = True)
    results = train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
    #import cProfile
    #cProfile.run('trainy()')
    #trainy()
