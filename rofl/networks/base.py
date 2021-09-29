from rofl.functions.const import *
from rofl.functions.functions import nn, no_grad

def isItem(T):
    if T.squeeze().shape == ():
        return True
    return False

def simpleActionProc(action, discrete):
    if discrete and isItem(action):
        action = action.item()
    else:
        action = action.to(DEVICE_DEFT).squeeze().numpy()
    return action

class BaseNet(nn.Module):
    """
    Class base for all the networks. Common methods.
    It has the following to configure:
    - discrete
    - name
    - device
    """
    name = "BaseNet"
    
    def __init__(self):
        self.discrete, self.__dvc__ = True, None
        super(BaseNet, self).__init__()

    def new(self):
        """
        This method must return the same architecture when called from a 
        already given network.
        """
        raise NotImplementedError

    @property
    def device(self):
        if self.__dvc__ is None:
            self.__dvc__ =  next(self.parameters()).device
        return self.__dvc__

class Value(BaseNet):
    """
    Class design to manage a observation value function only
    """
    name = "Value Base"
    def __init__(self):
        self.discrete = False
        super(Value,self).__init__()

    def getValue(self, x, action = None):
        with no_grad():
            value = self.forward(x)
        return value.item()

class QValue(BaseNet):
    """
    Class design to manage an action-value function
    """
    name = "QValue Base"
    def __init__(self):
        self.discrete = True
        super(QValue, self).__init__()
        
    def processAction(self, action):
        return simpleActionProc(action, self.discrete)

    def getQValues(self, observation):
        with no_grad():
            return self.forward(observation)

    def getValue(self, observation, action):
        assert isinstance(action, (int, list, tuple)), "action must be a int, list or tuple type"
        return self.getQValues(observation)[action].item()

    def getAction(self, observation):
        """
        Returns the max action from the Q network. Actions
        must be from 0 to n. That would be the network output
        """
        max_a = self.getQValues(observation).argmax(1)

        return self.processAction(max_a)

class Actor(BaseNet):
    """
    Class design to manage an actor only network
    """
    name = "Actor Base"
    def __init__(self):
        super(Actor, self).__init__()

    def onlyActor(self, observation):
        return self.forward(observation)

    def getDist(self, x):
        """
        From the actorForward, returns the corresponding pytorch distributions object to 
        sample the action from and to return .log_prob()
        """
        raise NotImplementedError

    def processAction(self, action):
        """
            Given the network properties, process the actions
            accordingly
        """
        return simpleActionProc(action, self.discrete)

    def getAction(self, observation):
        """
        From a tensor observation returns the sampled actions.

        returns
        -------
        action
        """
        with no_grad():
            dist = self.getDist(self.onlyActor(observation))
            action = dist.sample()
        return self.processAction(action)
        
    def sampleAction(self, params):
        """
        Creates, samples and returns the action and log_prob for it

        parameters
        ----------
        params:
            Raw output logits from the network

        returns
        action, log_prob, entropy
        """
        dist = self.getDist(params)
        action = dist.sample()
        log_prob, entropy = dist.log_prob(action), dist.entropy()
        return action, log_prob, entropy

class ActorCritic(Actor):
    """
    Class design to host both actor and critic for those architectures when a start 
    part is shared like a feature extraction from a CNN as for DQN-Atari.
    """
    name = "Actor critic Base"
    def __init__(self):
        super(ActorCritic, self).__init__()
        
    def sharedForward(self, x):
        """
        From the observation, extracts the features. Recomended to return the 
        flatten tensor in batch form
        """
        raise NotImplementedError

    def valueForward(self, x):
        """
        From the feature extraction. Calculates the value from said observation
        """
        raise NotImplementedError

    def actorForward(self, x):
        """
        From the feature extraction. Calculates the raw output to represent the parameters
        for the actions distribution.
        """
        raise NotImplementedError

    def onlyActor(self, observation):
        return self.actorForward(self.sharedForward(observation))

    def forward(self, x):
        features = self.sharedForward(x)
        values = self.valueForward(features)
        raw_actor = self.actorForward(features.clone())

        return values, raw_actor
    
    def getValue(self, x, action = None):
        """
        Form a tensor observation returns the value approximation 
        for it with no_grad operation.
        """
        with no_grad():
            value = self.getValues(x)
        return value.item()

    def getValues(self, observations):
        """
            Returns raw values from the critic part of the network.
        """
        return self.valueForward(self.sharedForward(observations))