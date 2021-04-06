from rofl.functions.const import *

class BaseNet(nn.Module):
    """
    Class base for all the networks. Common methods.
    It has the following to configure:
    - discrete
    - name
    - device
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.discrete = True
        self.__dvc__ = None
        self.name = "BaseNet"

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
    def __init__(self):
        super(Value,self).__init__()
        self.name = "ValueBase"

    def getValue(self, x):
        with no_grad():
            value = self.forward(x)
        return value.item()

class QValue(BaseNet):
    """
    Class design to manage an action-value function
    """
    def __init__(self):
        super(QValue, self).__init__()
        self.discrete = True
        self.name = "QValue"

    def getAction(self, x):
        """
        Returns the max action from the Q network. Actions
        must be from 0 to n. That would be the network output
        """
        with no_grad():
            values = self.forward(x)
            max_a = values.argmax(1)

        if values.shape[0] == 1:
            return max_a.item()
        else:
            return max_a.to(DEVICE_DEFT).squeeze().numpy()

class Actor(BaseNet):
    """
    Class design to manage an actor only network
    """
    def __init__(self):
        super(Actor, self).__init__()
        self.name = "ActorBase"

    def getDist(self, x):
        """
        From the actorForward, returns the corresponding pytorch distributions objecto to 
        sample the action from and to return .log_prob()
        """
        raise NotImplementedError

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
        return self.sampleAction(distParams)
    
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
        
        if self.discrete:
            action = action.item()
        else:
            action.to(DEVICE_DEFT).squeeze(0).numpy()

        return action, log_prob, entropy

class ActorCritic(Actor):
    """
    Class design to host both actor and critic for those architectures when a start 
    part is shared like a feature extraction from a CNN as for DQN-Atari.
    """
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.name = "Actor_critic"
        
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
        return self.sampleAction(distParams)

    def getValue(self, x):
        """
        Form a tensor observation returns the value approximation 
        for it with no_grad operation.
        """
        with torch.no_grad():
            value = self.valueForward(self.sharedForward(x))
        return value.item()

    