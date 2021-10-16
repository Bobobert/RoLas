from rofl.functions.torch import maxGrad, meanGrad
from rofl.networks.base import Value, QValue, Actor, ActorCritic
from rofl.functions.functions import nn, no_grad
from rofl.functions.const import DEVICE_DEFT
from rofl.utils.policies import getActionWProb, logProb4Action
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
        self.keysForUpdate = None

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
        if self.gae and not self.valueBased:
            raise Exception('%s cannot process GAE while not value based!' % self)

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
        if self.stochastic:
            return self.actor.getAction(observation)
        raise NotImplementedError

    def getActionWProb(self, observation, *kwargs):
        """
            New, in test
        """
        if not self.stochastic:
            raise AttributeError('%s does not support this operation')
        with no_grad():
            return getActionWProb(self.actor, observation)

    def getActionWVal(self, observation):
        """
            New, in test
            
            returns
            --------
            - action
            - value
        """
        if not self.valueBased:
            raise AttributeError('%s does not support this operation')
        action = self.getAction(observation)
        value = self.getValue(observation, action)
        return action, value

    def getAVP(self, observation):
        """
            New, in test
            From an observation returns all the policy can process about it:
            an action, a value and the log_prob of said action.

            returns
            --------
            - action
            - value of the observation
            - log_prob of action
        """
        if not self.valueBased and not self.stochastic:
            raise AttributeError('%s does not support this operation')
        with no_grad():
            action, logprob = self.getActionWProb(observation)
        value = self.getValue(observation, action)
        return action, value, logprob

    def getProb4Action(self, observation, action):
        """
            New in test
            returns
            -------
            - log_prob tensor for the action
        """
        return logProb4Action(self, observation, action)

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
        policyClass = self.__class__
        actor = self.actor
        if actor is not None:
            try:
                newActor = actor.new()
            except AttributeError:
                print('Warning: %s couldnt create a new actor' % self)
                newActor = None
        new = policyClass(self.config, newActor)
        return new

    def getActions(self, batchDict):
        """
            Batch mode for getAction method.

            returns
            -------
            actions in batch, ids in batch

        """
        observations = batchDict["observation"]
        actions = []
        for n in range(batchDict['N']):
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

    def _evalTBWActor_(self):
        tbw, actor, epoch = self.tbw, self.actor, self.epoch
        if self.evalMeanGrad:
            tbw.add_scalar("train/mean grad",  meanGrad(actor), epoch)
        if self.evalMaxGrad:
            tbw.add_scalar("train/max grad",  maxGrad(actor), epoch)
        
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
        self.keysForUpdate = None

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
