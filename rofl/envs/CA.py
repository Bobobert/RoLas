from .forestFire.helicopter import EnvMakerForestFire
from gym.spaces import Discrete
import numpy as np

def forestFireEnvMaker(config):
    config = config["env"]
    def ENV(seed = None):
        env = EnvMakerForestFire(n_row=config["n_row"], n_col = config["n_col"],
        p_tree = config.get("p_tree", 0.01), p_fire=config.get("p_fire", 0.005),
        ip_tree = config.get("ip_tree", 0.6), ip_fire = config.get("ip_fire", 0.0),
        ip_rock = 0.0, ip_lake = 0.0,
        observation_mode= config.get("obs_mode", "followGridImg"),
        observation_shape= config.get("obs_shape", (32,32)),
        reward_type = config.get("reward_type", "hit"), moves_before_updating=config.get("freeze", 4),
        steps_to_termination = config.get("steps_termination", 128),
        reward_move= config.get("reward_move", -0.001), reward_hit=config.get("reward_hit",0.01),
        reward_tree = config.get("reward_tree", 1.0), reward_fire = config.get("reward_fire", -1.0),
        reward_empty=0.0)
        env.rg = nprnd.Generator(nprnd.SFC64(seed))
        return env, [seed]

    # Register action_space on config
    config["env"]["action_space"] = Discrete(config["policy"].get("n_actions", 18))

    return ENV
