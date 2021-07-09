from rofl.policies.base import Policy
from rofl.functions.gym import noOpSample
from rofl.functions.const import *

class dummyPolicy(Policy):
    name = "DummyPlug"
    config = {}
    def __init__(self, env):
        super().__init__()
        self._as = env.action_space
        self.nop = noOpSample(env)

    def getAction(self, state):
        return self.nop

    def getRandom(self):
        return self._as.sample()

    @property
    def device(self):
        return DEVICE_DEFT