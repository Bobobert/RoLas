from gym.spaces import Space
from .const import ARRAY

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