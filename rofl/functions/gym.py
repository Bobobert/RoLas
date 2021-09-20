from gym.spaces import Space
from gym import Env
from .const import ARRAY, rnd

def noOpSample(env):
    """
        Returns a no-op for the environment. With the
        supposition this is always a zero.  
    """
    sample = env.action_space.sample()
    if isinstance(sample, (int)):
        return 0
    elif isinstance(sample, float):
        return 0.0
    elif isinstance(sample, ARRAY):
        return sample.fill(0.0)

def assertActionSpace(config):
    sus = config["env"].get("action_space")
    assert isinstance(sus, Space), "Space needs to be a Space from gym package"
    return sus

def someSkips(min_steps, max_steps):
    max_steps = 1 if max_steps < min_steps else max_steps
    min_steps = 0 if min_steps < 0 else min_steps
    return rnd.randint(min_steps, max_steps)

def skip(env, op, skips):
    obs = env.reset()
    for _ in range(skips):
        ac = op(obs)
        obs, r, done, _ = env.step(ac)
        if done:
            return False, None, ac
    return True, obs, ac

def warmUp(env, op, min_steps, max_steps):
    """
        Environment warmup. With n-steps op actions
        of the interval [min_step, max_steps].

        The function could be expensive if the n-steps are not
        feasible in the environment with the given operation.

    """
    TRIES = 5
    skips = someSkips()
    
    for _ in range(TRIES):
        achieved, obs, action = skip(env, op, skips)
        if achieved:
            return obs, skips, action
        skips += -1
    
    print("Warmup range [{}, {}] not feasible. Returned 0-step skip environemnt".format(
                                                                                        min_steps,
                                                                                        max_steps))
    return env.reset(), 0, noOpSample(env)

def noOpStart(env, min_steps = 0, max_steps = 30):
    noop = noOpSample(env)
    def nop(*any):
        return noop

    return warmUp(env, nop, min_steps, max_steps)

def randomStart(env, min_steps = 0, max_steps = 30):
    AS = env.action_space
    def randop(*any):
        return AS.sample()

    return warmUp(env, randop, min_steps, max_steps)

def doWarmup(warmup:str, env: Env, envConfig):
    """
        Switch case to call the warmups functions
        to start an environment.

        parameters
        ----------
        warmup: str
            Key for the warmup function
        env: Env
            A gym environment
        envConfig:
            Configuration dictionary for the environment
            parameters

        return
        ------
        observation, steps_excuted
    """
    mn = envConfig.get("warmup_min_steps", 0)
    mx = envConfig.get("warmup_max_steps", 0)
    if warmup == "noop":
        return noOpStart(env, mn, mx)
    elif warmup == "random":
        return randomStart(env, mn, mx)
    else:
        raise KeyError("Warmup key {} is not valid".format(warmup))
