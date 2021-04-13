from gym import make
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