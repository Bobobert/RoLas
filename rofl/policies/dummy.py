from rofl.policies.base import Policy
from rofl.functions.const import DEVICE_DEFT

class dummyPolicy(Policy):
    """
        Pilot's useless. Activate the dummy plug.

        Meant to be initialized by an agent.

        parameters
        ----------
        noOp: from a noOpSample of the target environment
    """
    name = "DummyPlug"
    config = {}
    def __init__(self, noOp):
        super().__init__()
        self.noop = noOp

    def getAction(self, observation):
        return self.noop

    @property
    def device(self):
        return DEVICE_DEFT

    def new(self):
        newDummy = dummyPolicy(self.noop)
        newDummy.rndFunc = self.rndFunc
        return newDummy

    def loadState(self, newState):
        pass

    def currentState(self):
        return dict()
