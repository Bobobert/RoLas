from rofl.networks.base import Value, QValue, Actor, ActorCritic
from abc import ABC


class Policy(ABC):
    """
    Base class for a Policy

    Must be initiated with:
    - A configuration dictionary
    Optionals:
    - Approximation functions
    - Exploratory strategy

    Methods
    -------
    - getAction: returns the action corresponding
        to the state given
    - getActions: batch mode for getAction()
    - exploreAgenda: means to manage the exploration of the policy for the
        getAction method 
    - getRndAction: returns a random action. Defaults uses an agent's
        rndAction() as rndFunc
    - getValue: if possible, will return the value for a state or
        pair state-action
    - sampleAction: returns the action, probability, and entropy
        from the action's distribution for actor based policies
    - update: Depending the type of policy updates itself.
        Inputs must be always dictionaries containing the update's
        information or material for approximations

    Other methods
    ----------
    - currentState: returns a dict
    - loadState: Loads a state dict
    """
    name = "BasePolicy"
    envName, config = None, None
    discrete, test = True, False
    exploratory, tbw = None, None
    actor, rndFunc = None, None

    def __init__(self):
        if self.name == "BasePolicy":
            raise NameError("New agent should be called different to BaseAgent")
        if self.config is None or not isinstance(self.config, dict):
            raise ValueError("Agent needs .config as a dict")

    def __call__(self, state):
        return self.getAction(state)
    
    def getAction(self, state, **kwargs):
        """
            From the given state, return an action
            as int if discrete or as a numpy.ndarray if
            continuous
        """
        raise NotImplementedError

    def getRndAction(self):
        """
            Returns an action from the expected
            action space.
        """
        if self.rndFunc is None:
            raise NotImplementedError
        return self.rndFunc()

    def exploreAgenda(self, state, **kwargs):
        """
            Intended to be invoked in getAction method, to manage the
            exploration agenda
        """
        pass

    def getValue(self, state, action = None):
        """
            Calculates from the actor object a value for a given state repsentation, if possible.
            
            returns
            --------
            float
        """
        actor = self.actor
        if actor is None:
            raise TypeError("Policy does not have any object under the alias actor, thus cannot calculate any value")
        if isinstance(actor, Actor):
            raise TypeError("Policy's actor is an Actor network type, thus cannot calculate any value")
        if isinstance(actor, (Value, ActorCritic)):
            return actor.getValue(state)
        if isinstance(actor, QValue):
            assert action != None, "The actor alias points to a q-network, please provide the action argument"
            return actor.getValue(state, action)

    def sampleAction(self, state):
        """
            Return the raw action, log_prob and entropy
            from the action's distribution of the actor
            policy.
        """
        if self.actor != None and isinstance(self.actor, (Actor, ActorCritic)):
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
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the policy.
            Must verify the name of the policy is the same and the
            type.
        """
        raise NotImplementedError

    def registerAgent(self, agent):
        """
            If required, a policy can or should register an Agent
            to reference some methods; eg. rndAction(), or attributes.
        """
        self.rndFunc = agent.rndAction
        self.envName = agent.envName

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

    def getActions(self, batchDict):
        """
            Batch mode for getAction method.

            returns
            -------
            actions in batch, ids in batch

        """
        observations = batchDict["observation"]
        N = observations.shape[0]
        actions = []
        for n in range(N):
            actions.append(self.getAction(observations[n].unsqueeze(0)))

        return actions, batchDict["id"]


    