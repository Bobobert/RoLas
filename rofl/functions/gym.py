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