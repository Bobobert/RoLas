from rofl.networks.base import Value, QValue, Actor, ActorCritic
import torch.nn as nn
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
    config, discrete, test = {}, None, False
    exploratory, tbw, tbwFreq = None, None, None
    actor, rndFunc, valueBased = None, None, None
    gamma, lmbd, gae = 1.0, 1.0, False

    def __init__(self, config, actor, **kwargs):
        if self.name == "BasePolicy":
            raise ValueError("New agent should be called different to BasePolicy")
        
        if config is None or not isinstance(config, dict):
            raise ValueError("Agent needs config as a dict")

        self.config = config
        self.actor = actor
        self.tbw = kwargs.get('tbw')
        self.tbwFreq = config['policy']['evaluate_freq']

        self.gamma, self.lmbd = config['agent']['gamma'], config['agent']['lambda']
        self.evalMaxGrad = config['policy']["evaluate_max_grad"]
        self.evalMeanGrad = config['policy']["evaluate_mean_grad"]
        self.clipGrad = config['policy']['clip_grad']

        self.initPolicy(**kwargs)
        self.__checkInit__()

    def __checkInit__(self):
        # testing for valueBased, required for agent class working for any value method
        if isinstance(self.actor, (Actor)):
            self.valueBased = False
        if isinstance(self.actor, (Value, QValue, ActorCritic)):
            self.valueBased = True
        if self.valueBased is None: # as ActorCritic is based on Actor, those will be tested twice
                raise ValueError("Attribute .valueBased should be declared to a boolean type")
        if self.discrete is None:
            try:
                self.discrete = self.actor.discrete
            except AttributeError:
                raise ValueError("Attribute .discrete should be declared to a boolean type")

    def initPolicy(self, **kwargs):
        """
            If needed, write additional initialization for 
            parameters and functions setup.

            returns
            -------
            None
            
        """
        pass

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
        if self.valueBased:
            return self.actor.getValue(state, action)
        raise TypeError("{} is not value based, thus cannot calculate any value".format(self.name))

    def sampleAction(self, state):
        """
            Return the raw action, log_prob and entropy
            from the action's distribution of the actor
            network.
        """
        if isinstance(self.actor, Actor):
            return self.actor.sampleAction(state)
        raise TypeError("{} does not have an Actor type as .actor".format(self.name))
        
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
        if isinstance(self.actor, nn.Module):
            return self.actor.device
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


    