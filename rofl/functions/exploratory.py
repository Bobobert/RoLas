from .vars import linearSchedule
from .const import assertProb

class EpsilonGreedy():
    """
        Schedule for epsilon-greedy

        Modes supported:
        - "linear"
    """
    def __init__(self, initial, last, life, mode, test = None):
        
        assertProb(initial), assertProb(last), assertProb(test)

        if mode == "linear":
            dec = initial > last
            self._var_ = linearSchedule(initial, life, minValue= last if dec else None,
                                        maxValue = None if dec else last)
        else:
            raise NotImplementedError
        
        self._test_ = test if test is not None else last
        
    def train(self, obs):
        return self._var_

    def test(self, obs):
        return self._test_

    def reset(self):
        self._var_.restore()