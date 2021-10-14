from rofl.config import createConfig, createAgent, getEnvMaker
from rofl.algorithms.a2c import algConfig
from rofl.config.config import createNetwork, createPolicy

envConfig = {
    'envMaker' : 'gymEnvMaker',
    'name': 'LunarLanderContinuous-v2',
    'atari': False,
    'max_length': 500,
    'warmup' : None,
    }

agentCnf = {
    'agentClass' : 'agentMaster',
}
policyCnf = {
    'policyClass' : 'pgPolicy',
    'continuos' : True,
    'network' : {
        'networkClass' : 'gymActor',
        'linear_1' : 56,
    }
}

thisCnf = {
    'env': envConfig,
    'policy' : policyCnf,
    #'agent' : agentCnf,
}
if __name__ == '__main__':
    config = createConfig(algConfig, thisCnf, expName='a2c')

    eMaker = getEnvMaker(config)
    actor = createNetwork(config)
    policy = createPolicy(config, actor)
    agent = createAgent(config, policy, eMaker)
    try:
        for _ in range(10 ** 6):
            agent.fullStep()
    except KeyboardInterrupt:
        print('Done by keyboard...')
    print('oks')
    agent.close()
