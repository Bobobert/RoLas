from rofl.functions.torch import maxGrad, meanGrad
from rofl.networks.base import Value, QValue, Actor, ActorCritic
from rofl.functions.functions import nn, no_grad
from rofl.functions.const import DEVICE_DEFT
from abc import ABC

class Policy(ABC):
    """
    Base class for a Policy

    Must be initiated with:
    - A configuration dictionary

    Methods
    -------
    - initPolicy: the additional initialization for custom policies.
        in this call other networks besides actor('network') should be created.
    - getAction: returns the action corresponding
        to the observation given
    - getActions: batch mode for getAction()
    - explorationAgenda: means to manage the exploration of the policy for the
        getAction method 
    - getRndAction: returns a random action. Defaults uses an agent's
        rndAction() as rndFunc
    - getValue: if possible, will return the value for a observation or
        pair observation-action
    - update: Depending the type of policy updates itself.
        Inputs must be always dictionaries containing the update's
        information or material for approximations

    Properties
    ---------
    - test: Sets the flags for the policy to behave in a test if any
        difference.
    - train: Same as test, but does the oposite

    Other methods
    ----------
    - currentState: returns a dict
    - loadState: Loads a observation dict
    """
    name = "BasePolicy"

    def __init__(self, config, actor, **kwargs):

        self.discrete, self._test = None, False
        self.exploratory, self.epoch = None, 0
        self.rndFunc, self.valueBased = None, None
        self.stochastic, self._nn, self._sharedNN =  False, False, False

        if self.name == "BasePolicy":
            raise ValueError("New agent should be called different to BasePolicy")
        
        if config is None or not isinstance(config, dict):
            raise ValueError("Agent needs config as a dict")
        
        self.config = config
        self.actor = actor
        self.tbw = kwargs.get('tbw')
        self.tbwFreq = config['policy']['evaluate_tb_freq']

        self.gamma, self.lmbd = config['agent']['gamma'], config['agent']['lambda']
        self.gae = config['agent']['gae']
        self.evalMaxGrad = config['policy']["evaluate_max_grad"]
        self.evalMeanGrad = config['policy']["evaluate_mean_grad"]
        self.clipGrad = config['policy']['clip_grad']

        self.initPolicy(**kwargs)
        self.__checkInit__()

    def __checkInit__(self):
        # testing for valueBased, required for agent class working for any value method
        if isinstance(self.actor, (Actor)):
            self.stochastic = True
        if isinstance(self.actor, (Value, QValue, ActorCritic)):
            self.valueBased = True
        if self.valueBased is None: # as ActorCritic is based on Actor, those will be tested twice
                raise ValueError("Attribute .valueBased should be declared to a boolean type")
        if self.discrete is None:
            try:
                self.discrete = self.actor.discrete
            except AttributeError:
                raise ValueError("Attribute .discrete should be declared to a boolean type")
        if isinstance(self.actor, nn.Module):
            self._nn = True

    def initPolicy(self, **kwargs):
        """
            If needed, write additional initialization for 
            parameters and functions setup.

            returns
            -------
            None
            
        """
        pass

    def __call__(self, observation):
        return self.getAction(observation)
    
    def getAction(self, observation, **kwargs):
        """
            From the given observation, return an action
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

    def explorationAgenda(self, observation, **kwargs):
        """
            Intended to be invoked in getAction method, to manage the
            exploration agenda
        """
        pass

    def getValue(self, observation, action):
        """
            Calculates from the actor object a value for a given observation repsentation, if possible.
            
            returns
            --------
            float
        """
        if self.valueBased:
            return self.actor.getValue(observation, action)
        raise TypeError("{} is not value based, thus cannot calculate any value".format(self.name))
        
    def update(self, batchDict):
        """
            From the information dictionaries,
            the policy should be updated. If tabular
            make the incremental or new observation.
            If using approximations, this would have
            to contain the information to update the 
            parameters. Meaning the optimizer should be given 
            here adhoc to the policy.
        """
        raise NotImplementedError

    def currentState(self):
        """
            Returns a dict with all the required information
            of its observation to start over or just to save it.
        """
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary observation, loads all the values into
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
        if self._nn:
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

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, flag:bool):
        if not isinstance(flag, bool):
            raise ValueError("flag needs to be a boolean type, not {}".format(type(flag)))
        self._test = flag
        if self._nn and flag:
            self.actor.eval()
        elif self._nn and not flag:
            self.actor.train()

    @property
    def train(self):
        return not self._test
    
    @train.setter
    def train(self, flag: bool):
        self.test = not flag

    def shareMem(self):
        if self._nn and self.device == DEVICE_DEFT:
            self._sharedMem = True
            self.actor.share_memory()

    def _evalTBWActor_(self):
        if self.evalMeanGrad:
            self.tbw.add_scalar("train/mean grad",  meanGrad(self.actor), self.epoch)
        if self.evalMaxGrad:
            self.tbw.add_scalar("train/max grad",  maxGrad(self.actor), self.epoch)
        
class dummyPolicy(Policy):
    """
        Pilot's useless. Activate the dummy plug.

        Meant to be initialized by an agent.

        parameters
        ----------
        noOp: from a noOpSample of the target environment
    """
    name= "DummyPlug"
    def __init__(self, noOp):
        self.noop = noOp
        self.discrete = True
        self.valueBased = False
        self.stochastic = False
        self.config = {}
        self._test = False

    def getAction(self, observation):
        return self.noop

    def getActions(self, batchDict):
        return super().getActions(batchDict)

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
