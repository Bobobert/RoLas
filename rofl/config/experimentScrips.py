from rofl.policies.base import Policy
from rofl.functions.torch import getDevice
from rofl.config.config import createConfig, getEnvMaker, getTrainFun,\
    createNetwork, createAgent, createPolicy
from rofl.utils.random import seeder
from rofl.utils import pathManager

def setUpExperiment(algorithm: str, *configs : dict, dummyManager : bool = False,
                        cuda: bool = True, seed: int = 8080):
    """
        To be called in main

        parameters
        ----------
        - algorithm: str
            name of the algorithm and its python module
        - configs: dict
            all the dicts to complement the config dict
        - dummyManager: bool
            Default False. Initialices the pathManager in dummy or
            normal mode.
        - cuda: bool
            Default True. Try to initialice pytorch and manage
            the networks in a cuda capable device.
        - seed: int
            Default 8080. Seeder for the three main random number
            libraries: random, numpy.random, torch.random
        
        returns
        -------
        - config: dict
        - agent : Agent type
        - policy: Policy Type
        - train function: function
        - manager: pathManager
    """
    device = getDevice(cudaTry = cuda)
    seeder(seed, device)
    
    config = createConfig(*configs, expName = algorithm)
    config['seed'] = seed
    manager = pathManager(config, dummy = dummyManager)
    experiment = startExperiment(config, manager, device)
    manager.saveConfig()
    print('The experiment is ready, time for %s algorithm in %s environment' % (algorithm, config['env']['name']))
    return experiment

def startExperiment(config, manager, device):
    writer = manager.startTBW()
    envMaker = getEnvMaker(config)
    actor = createNetwork(config, key = 'actor').to(device) # TODO: perhaps, another options besides a network!
    policy = createPolicy(config, actor, device = device, tbw = writer)
    agent = createAgent(config, policy, envMaker, device = device, tbw = writer)
    train = getTrainFun(config)
    return config, agent, policy, train, manager

def loadExperiment(algorithm: str, environment: str,
                        cuda: bool = True, seed: int = 8081):
    """
        To be called in main

        parameters
        ----------
        - algorithm: str
            name of the algorithm used
        - environment: str
            name of the environment used
        - dummyManager: bool
            Default True. Initialices the pathManager in dummy or
            normal mode.
        - cuda: bool
            Default True. Try to initialice pytorch and manage
            the networks in a cuda capable device.
        - seed: int
            Default 8080. Seeder for the three main random number
            libraries: random, numpy.random, torch.random
        
        returns
        -------
        - config: dict
        - agent : Agent type
        - policy: Policy Type
        - train function: function
        - manager: pathManager
    """
    device = getDevice(cudaTry = cuda)
    seeder(seed, device)
    config = createConfig(expName = algorithm)
    config['env']['name'] = environment
    manager = pathManager(config, dummy = False, load = True)
    config = manager.config
    assertVerions()
    manager.dummy = True # TODO; this could change later, when getState and loadState work
    experiment = startExperiment(config, manager, device)
    return experiment

def assertVerions():pass

def createGifFromAgent(steps: int = 100, actionsPerSec: int = 4): # TODO; set policy to 
    # use DQN.Trainer.playTest as reference
    return
    from tqdm import tqdm
    try:
        import gif
    except ImportError:
        gif = None
    Policy.test = True
    frames = []
    for _ in tqdm(range(steps), desc = 'Playing policy', unit='fullStep'):
        agent.fullStep()
        render()
        frames.append(frame) if gif else None
        doPause()
    return
