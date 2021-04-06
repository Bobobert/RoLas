from rofl.functions.const import *
from rofl.functions.torch import *
from abc import ABC

class Policy(ABC):
    """
    Base class for a Policy

    Must be initiated with:
    - A configuration dictionary
    Optionals:
    - Approximation functions
    - Exploratory strategy

    Methods:
    - getAction: returns the action corresponding
        to the state given
    - update: Depending the type of policy updates itself.
        Inputs must be always dictionaries containing the update's
        information or material for approximations
    - currentState: returns a dict
    - loadState: Loads a state dict
    - device: property if enable, otherwise None
    """
    name = "BasePolicy"
    environment, config = None, None
    discrete, test = True, False
    exploratory, tbw = None, None
    def __init__(self):
        if self.name == "BasePolicy":
            raise NameError("New agent should be called different to BaseAgent")
        if self.config is None or not isinstance(self.config, dict):
            raise ValueError("Agent needs .config as a dict")

    def __call__(self, state):
        return self.getAction(state)
    
    def getAction(self, state):
        """
            From the given state, return an action
            as int if discrete or as a numpy.ndarray if
            continuous
        """
        raise NotImplementedError

    def update(self, *infoDicts):
        """
            From the information dictionaries,
            the policy should be updated. If tabular
            make the incremental or new state.
            If using approximations, this would have
            to contain the information to update the 
            parameters. Meaning the optimizer should be given 
            here adhoc to the policy.
        """
        raise NotImplementedError

    def currentState(self):
        """
            Returns a dict with all the required information
            of its state to start over or just to save it.
        """
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the policy.
            Must verify the name of the policy is the same and the
            type.
        """
        return NotImplementedError

    @property
    def device(self):
        return None

    def __repr__(self):
        s = "Policy {}\nFor environment {}\nDiscrete {}".format(self.name, 
            self.environment, self.discrete)
        return s
