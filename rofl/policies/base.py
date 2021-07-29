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
    - getRandom: returns a sample from the action space
    - sampleAction: returns the action, probability, and entropy
        from the action's distribution for actor based policies.
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
    actor = None

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

    def getRandom(self):
        """
            Returns an action from the expected
            action space. If there is an exploratory
            agenda. This should not interact with it.
        """
        raise NotImplementedError

    def sampleAction(self, state):
        """
            Return the raw action, log_prob and entropy
            from the action's distribution of the actor
            policy.
        """
        if self.actor != None:
            return self.actor.sampleAction(state)
        return None
        
    def update(self, batchDict):
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
        raise NotImplementedError

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the policy.
            Must verify the name of the policy is the same and the
            type.
        """
        raise NotImplementedError

    @property
    def device(self):
        return None

    def __repr__(self):
        s = "Policy {}\nDiscrete {}".format(self.name, self.discrete)
        return s

    def new(self):
        """
            Returns a new policy based on the same definitions and 
            parameters as the one called from. This method should also create
            new copies of networks.

            returns
            -------
            policy
        """
        raise NotImplementedError

    def getActions(self, infoDict):
        """
            Batch mode for getAction method.

            returns
            -------
            actions in batch, ids in batch

        """
        observations = infoDict["observation"]
        N = observations.shape[0]
        actions = []
        for n in range(N):
            actions.append(self.getAction(observations[n].unsqueeze(0)))

        return actions, infoDict["id"]


    