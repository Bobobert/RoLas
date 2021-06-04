from gym import make
import gym_cellular_automata as gymca

try:
    from gym.envs.atari import AtariEnv
except:
    print("Atari Environments are not available")
    AtariEnv = None

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
        env = AtariEnv(name, obs_type = "image", frameskip = config["env"].get("frameskip", 4))
        seeds = env.seed(seed)
        return env, seeds

    return ENV

def gymcaEnvMaker(config):
    name = config["env"]["name"]
    if name not in gymca.REGISTERED_CA_ENVS:
        raise ValueError(
            "Environment {} is not registered in gym_cellular_automata".format(name))

    def ENV(seed = None):
        env = make("gym_cellular_automata:" + name)
        env.seed(seed)
        return env, [seed]

    return ENV
