from .forestFire.helicopter import EnvMakerForestFire
from .gym_cellular_automata.envs.bulldozer import BulldozerEnv
from .gym_cellular_automata.envs.forest_fire import ForestFireEnv
import numpy as np

def forestFireEnvMaker(config):
    config = config["env"]
    def ENV(seed = None):
        env = EnvMakerForestFire(n_row=config["n_row"], n_col = config["n_col"],
        p_tree = config.get("p_tree", 0.01), p_fire=config.get("p_fire", 0.005),
        ip_tree = config.get("ip_tree", 0.6), ip_fire = config.get("ip_fire", 0.0),
        observation_mode= config.get("obs_mode", "followGridImg"),
        observation_shape= config.get("obs_shape", (32,32)),
        reward_type = config.get("reward_type", "hit"), moves_before_updating=config.get("freeze", 4))
        env.rg = np.random.Generator(np.random.SFC64(seed))
        return env, [seed]

    return ENV

def bulldozerEnvMaker(config):
    def ENV(seed = None):
        env = BulldozerEnv()
        env.seed(seed)
        return env, [seed]
        
    return ENV
    