from rofl.functions.const import *

def isItem(T):
    if T.squeeze().shape == ():
        return True
    return False

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
        super(BaseNet, self).__init__()
        self.discrete = True
        self.__dvc__ = None

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
    Class design to manage a state value function only
    """
    name = "Value Base"
    def __init__(self):
        super(Value,self).__init__()

    def getValue(self, x):
        with no_grad():
            value = self.forward(x)
        return value.item()

class QValue(BaseNet):
    """
    Class design to manage an action-value function
    """
    name = "QValue Base"
    def __init__(self):
        super(QValue, self).__init__()
        self.discrete = True
        
    def processAction(self, action):
        if isItem(action):
            return action.item()
        else:
            return action.to(DEVICE_DEFT).squeeze().numpy()

    def getAction(self, x):
        """
        Returns the max action from the Q network. Actions
        must be from 0 to n. That would be the network output
        """
        with no_grad():
            values = self.forward(x)
            max_a = values.argmax(1)

        return self.processAction(max_a)

class Actor(BaseNet):
    """
    Class design to manage an actor only network
    """
    name = "Actor Base"
    def __init__(self):
        super(Actor, self).__init__()

    def getDist(self, x):
        """
        From the actorForward, returns the corresponding pytorch distributions objecto to 
        sample the action from and to return .log_prob()
        """
        raise NotImplementedError

    def processAction(self, action):
        """
            Given the network properties, process the actions
            accordingly
        """
        if self.discrete and isItem(action):
            action = action.item()
        else:
            action = action.to(DEVICE_DEFT).squeeze().numpy()
        return action

    def getAction(self, x):
        """
        From a tensor observation returns the sampled actions and 
        their corresponding log_probs from the distribution.

        returns
        -------
        action, log_prob, entropy
        """
        with no_grad():
            distParams = self.forward(x)
            action, _, _ = self.sampleAction(distParams)
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
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

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

    def forward(self, x):
        features = self.sharedForward(x)
        values = self.valueForward(features)
        raw_actor = self.actorForward(features.clone())

        return values, raw_actor

    def getAction(self, x):
        """
        From a tensor observation returns the sampled actions and 
        their corresponding log_probs from the distribution.

        returns
        -------
        action, log_prob, entropy
        """
        with torch.no_grad():
            distParams = self.actorForward(self.sharedForward(x))
            action, _, _ = self.sampleAction(distParams)
        return self.processAction(action)

    def getValue(self, x):
        """
        Form a tensor observation returns the value approximation 
        for it with no_grad operation.
        """
        with torch.no_grad():
            value = self.valueForward(self.sharedForward(x))
        return value.item()

    