from .vars import linearSchedule
from .const import assertProb, math, torch

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
            self._var_ = linearSchedule(c["epsilon_start"], c["epsilon_end"], c["epsilon_life"])
        
        self._test_ = c.get("epsilon_test", 0.0)
        
    def train(self, obs):
        return self._var_.value

    def test(self, obs):
        return self._test_

    def reset(self):
        self._var_.restore()

def qUCBAction(agent, t, nt, c, eps = 1e-4):
    """
            Consider nt as the array shape [n_actions,], that holds
            how many times the action have been seen at time t.

            parameters
            ----------
            t: int
                step relative to the Tensor nt
            nt: Tensor
                Should have how manytimes an action has been seen
                at the time t
    """
    qvalues = agent.getQvalues(agent.lastObs)
    lnt = qvalues.new_empty(qvalues.shape).fill_(math.log(t))
    qs = torch.addcdiv(qvalues, c, lnt, nt + eps)

    return qs.argmax().item()