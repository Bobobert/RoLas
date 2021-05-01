from .vars import linearSchedule
from .const import assertProb

class EpsilonGreedy():
    """
        Schedule for epsilon-greedy

        Modes supported:
        - "linear"
    """
    def __init__(self, config):
        c = config["policy"]
        if c.get("epsilon") is not None:
            self._var_ = c["epsilon"]
        else:
            #Legacy
            self._var_ = linearSchedule(c["epsilon_start"], c["epsilon_life"], c["epsilon_end"])
        
        self._test_ = c.get("epsilon_test", 0.0)
        
    def train(self, obs):
        return self._var_.value

    def test(self, obs):
        return self._test_

    def reset(self):
        self._var_.restore()