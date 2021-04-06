from gym import make
from gym.envs.atari import AtariEnv

def gymEnvMaker(config):
    name = config["env"]["name"]
    def ENV(seed = None):
        if seed is not None and seed < 0:
            seed = None
        env = make(name)
        seeds = env.seed(seed)
        return env, seeds

    return ENV

def atariEnvMaker(config):
    name = config["env"]["name"]
    def ENV(seed = None):
        if seed is not None and seed < 0:
            seed = None
        env = AtariEnv(name, obs_type = "image")
        seeds = env.seed(seed)
        return env, seeds

    return ENV