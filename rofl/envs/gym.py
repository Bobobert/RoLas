from gym import make
from gym.spaces import Discrete
import gym_cellular_automata as gymca

try:
    from gym.envs.atari import AtariEnv
except:
    print("Atari Environments are not available")
    AtariEnv = None

def gymEnvMaker(config):
    """
    Standard gym environment maker.
    From config-env needs:
        - name
    
    returns a envMaker function
    """
    name = config["env"]["name"]
    def ENV(seed = None):
        if seed is not None and seed < 0:
            seed = None
        env = make(name)
        seeds = env.seed(seed)
        return env, seeds

     # Register action_space on config
    env, _ = ENV()
    config["env"]["action_space"] = env.action_space
    config['env']['observation_space'] = env.observation_space
    del env

    return ENV

def atariEnvMaker(config):
    """
    Standard gym atari environment maker
    From config-env needs:
        - name

    Optionals:
        - obsType
        - mode
        - difficulty
        - frameskip

    returns a envMaker function
    """
    name = config["env"]["name"]
    def ENV(seed = None):
        if seed is not None and seed < 0:
            seed = None
        env = AtariEnv(name, 
                        obs_type = config["env"].get("obsType", "image"),
                        mode = config["env"].get("mode", None),
                        difficulty= config["env"].get("difficulty", None),
                        frameskip = config["env"].get("frameskip", (2,5)),
                        )
        seeds = env.seed(seed)
        return env, seeds

    # Register action_space on config
    config["env"]["action_space"] = Discrete(config["policy"].get("n_actions", 18))

    return ENV

def gymcaEnvMaker(config):
    """
    Standard gym-cellular-automata environment
    Initialization is custom from the initWindKernel

    From config-env needs:
        - name
        
    Optionals:
        - n_row
        - n_col
        - wind_speed
        - wind_constant
        
    returns a envMaker function
    """
    name = config["env"]["name"]

    if name not in gymca.REGISTERED_CA_ENVS:
        raise ValueError(
            "Environment {} is not registered in gym_cellular_automata".format(name))

    DEFAULT_WIND_IMPORTANCE = 10
    # Config variables
    col, row = config["env"].get("n_col", 50), config["env"].get("n_row", 50)
    wind_params = (config["env"].get("wind_direction", 45),
                    config["env"].get("wind_speed", 10),
                    config["env"].get("wind_constat", DEFAULT_WIND_IMPORTANCE))
    # init function
    from .bulldozerUtils import initWindKernel

    def ENV(seed = None, wind_params = wind_params):
        env = make("gym_cellular_automata:" + name)
        # Override the .yaml paarmeters
        env._col = col
        env._row = row
        env._wind = initWindKernel(*wind_params)
        env.seed(seed)
        return env, [seed]

    # Register action_space on config
    env, _ = ENV()
    config["env"]["action_space"] = env.action_space
    del env

    return ENV
