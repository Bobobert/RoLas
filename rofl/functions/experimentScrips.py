from rofl.policies.base import Policy
from rofl.functions.torch import getDevice
from rofl.functions.config import createConfig
from rofl.utils.random import seeder
from rofl.utils import pathManager

def setUpExperiment(algorithm: str, *configs : dict, dummyManager : bool = True,
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
        - manager: pathManager
        - train function: function
    """
    device = getDevice(cudaTry = cuda)
    seeder(seed)
    config = createConfig(*configs, expName = algorithm)
    manager = pathManager(config, dummy = dummyManager)
    writer = manager.startTBW()

    manager.saveConfig()
    return config, pathManager

def createGifFromAgent(steps: int = 100, actionsPerSec: int = 4): # TODO; set policy to 
    # use DQN.Trainer.playTest as reference
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
