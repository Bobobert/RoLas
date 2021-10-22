from rofl.functions.const import *
from typing import Tuple
from rofl.functions.functions import multiplyIter, nn, sqrConvDim, no_grad, isItem, isBatch
from rofl.functions.torch import newNet

class BaseNet(nn.Module):
    """
    Class base for all the networks. Common methods.
    It has the following to configure:
    - discrete
    - name
    - config
    - device
    """
    name = "BaseNet"

    def __init__(self, config):
        self.discrete, self.__dvc__ = True, None
        self.config = config
        super(BaseNet, self).__init__()

    def new(self):
        """
        Returns a new instance of the class network with
        the actual stored config dictionary. If changes need to be
        made, use function newNet from a different config.

        returns
        -------
        new instance of network
        """
        return newNet(self, self.config)

    @property
    def device(self):
        if self.__dvc__ is None:
            self.__dvc__ =  next(self.parameters()).device
        return self.__dvc__

class Value(BaseNet):
    """
    Class design to manage a observation value function only

    methods
    -------
    - getValue: float or int
        From a pair observation and action, this net should output
        a single value. Action may not be required in the process.
        Uses no_grad

    """
    name = "Value Base"
    def __init__(self, config):
        super().__init__(config)
        self.discrete = False
    
    @no_grad()
    def getValue(self, observation, action):
        if isBatch(observation):
            return self.forward(observation).cpu().numpy()
        value = self.forward(observation)
        return value.item()

class QValue(BaseNet):
    """
    Class design to manage an action-value function

    methods
    -------
    - getValue:
        From a pair observation and action, this net should output
        a single value. Action may not be required in the process.
        Uses no_grad.
    - processAction:
        The output of the net could represent something different from
        what the problems needs, as this could be a great order of difference
        is up to the network to output a proper action, instead of a env wrapper
        to translate it.
    - unprocessAction:
        The inverse of processAction to interact again with tensor objects
    - getAction:
        From a single observation outputs a single action corresponding to the 
        maximum q_value given the observation. Uses no_grad.
    """
    name = "QValue Base"
    def __init__(self, config):
        super().__init__(config)
        self.discrete = True
    
    @no_grad()
    def getValue(self, observation, action):
        isbatch = isBatch(observation)
        action = self.unprocessAction(action, isbatch)

        if isbatch:
            Qvalues = self.forward(observation)
            values = Qvalues.gather(1, action)
            return values.cpu().numpy()

        return self.forward(observation)[action].item()
        
    def processAction(self, action):
        return simpleActionProc(action, self.discrete)

    def unprocessAction(self, action, batch: bool):
        if not batch:
            assert isinstance(action, (int, list, tuple)), "action must be a int, list or tuple type"
            return action
        return simpleActionUnProc(action, self.device)

    @no_grad()
    def getAction(self, observation):
        """
        Returns the max action from the Q network. Actions
        must be from 0 to n. That would be the network output
        """
        max_a = self.forward(observation).argmax(1)
        return self.processAction(max_a)

class Actor(BaseNet):
    """
    Class design to manage an actor only network

    methods
    -------
    - onlyActor: forward for the all the part of the architecture
        needed only to output an action
    - getDist: from a series of parameters, creates a distribution
        corresponding to the type of stochastic actor.
    - processDist: Needed to make easier policies, each type of actor
        should be capable of create, sample and process log prob and 
        entropies from its dristribution type.
    - processAction:
        The output of the net could represent something different from
        what the problems needs, as this could be a great order of difference
        is up to the network to output a proper action, instead of a env wrapper
        to translate it.
    - unprocessAction:
        Inverse of processAction to interact again with tensor objects.
    - getAction:
        From a single observation outputs a single action. Uses no_grad.
    """
    name = "Actor Base"
    def __init__(self, config):
        super().__init__(config)

    def onlyActor(self, observation):
        return self.forward(observation)

    def getDist(self, params):
        """
        From the actorForward, returns the corresponding pytorch distributions object
        which can be used to sample actions and their probabilities.

        parameters
        ----------
        - params: Tensor

        returns
        --------
        torch Distribution
        """
        raise NotImplementedError

    def processDist(self, params, actions):
        """
        More methods.. why not?!
        As some distributions treat diffently the tensor of actions, the results
        from log_probs() can be not as expected, resulting in a greater loss (when using policy grad).

        This is a method to treat the action batch for that the distribution output
        is a expected, i.e. [N, actions_sampled]. Eg. a batch of 10 actions from a 
        two normal dist should ouput a log_prob().shape = entropy().shape = [10, 2]. 

        Do not use no_grad() inside.

        parameters
        ----------
        - params: tensor
            The output of the actor network
        - actions: tensor
            Batch of actions, first dimension must match the params one.
            To be treated in the method.

        returns
        -------
        - log_probs: Tensor
        - entropies: Tensor
        """
        actions = self.unprocessAction(actions, isBatch(params))
        dist = self.getDist(params)

        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, entropies

    def processAction(self, action):
        """
            Given the network properties, process the actions
            accordingly
        """
        return simpleActionProc(action, self.discrete)

    def unprocessAction(self, action, batch: bool):
        """
            Should be the inverse of processAction, from which the
            actions can be utilize in distributions or to gather values.
        """
        actions = simpleActionUnProc(action, self.device)
        if self.discrete:
            actions = actions.squeeze()
        return actions

    @no_grad()
    def getAction(self, observation):
        """
        From a tensor observation returns the sampled actions.

        returns
        -------
        action
        """
        dist = self.getDist(self.onlyActor(observation))
        action = dist.sample()
        return self.processAction(action)

class ActorCritic(Actor):
    """
    Class design to host both actor and critic for those architectures 
    when a start part is shared, eg. feature extraction from a CNN.

    methods
    -------
    - sharedForward: 
        If there is a shared part of the architecture between the actor and the critic
        should be declared here how.
    - valueForward:
        Exclusive parts for the critic to output values.
    - actorForward:
        Exclusive parts for the actor to output parameters to create
        a distribution.
    - onlyActor: forward for all the parts of the architecture
        needed only to output an action
    - onlyValue : forwad for all the parts of the arch needed only
        to ouput a value from the critic.
    - getDist: from a series of parameters, creates a distribution
        corresponding to the type of stochastic actor.
    - processDist: Needed to make easier policies, each type of actor
        should be capable of create, sample and process log prob and 
        entropies from its dristribution type.
    - processAction:
        The output of the net could represent something different from
        what the problems needs, as this could be a great order of difference
        is up to the network to output a proper action, instead of a env wrapper
        to translate it.
    - getAction:
        From a single observation outputs a single action. Uses no_grad.
    - getValue:
        From a pair observation and action, this net should output
        a single value. Action may not be required in the process.
        Uses no_grad.
    """
    name = "Actor critic Base"
    def __init__(self, config):
        super().__init__(config)
        
    def sharedForward(self, observation):
        """
        From the observation, extracts the features. Recomended to return the 
        flatten tensor in batch form

        returns
        -------
        - Tensor
        """
        raise NotImplementedError

    def valueForward(self, observation):
        """
        From the feature extraction. Calculates the value from said observation
        
        returns
        --------
        - values
        """
        raise NotImplementedError

    def actorForward(self, observation):
        """
        From the feature extraction. Calculates the raw output to represent the parameters
        for the actions distribution.

        returns
        --------
        - parameters for dist
        """
        raise NotImplementedError

    def onlyActor(self, observation):
        return self.actorForward(self.sharedForward(observation))

    def onlyValue(self, observations):
        """
            Returns raw values from the critic part of the network.
        """
        return self.valueForward(self.sharedForward(observations))

    def forward(self, observation):
        """
        returns
        --------
        (values, params)

        - values: Tensor of values from the critic
        - params: Tensor of parameters for a distribution
        """
        features = self.sharedForward(observation)
        values = self.valueForward(features)
        raw_actor = self.actorForward(features.clone())

        return values, raw_actor
    
    @no_grad()
    def getValue(self, observation, action):
        """
        Form a tensor observation returns the value approximation 
        for it with no_grad operation.
        """
        value = self.onlyValue(observation)
        if isBatch(value):
            value = value.squeeze()
            return value.cpu().numpy()
        return value.item()

### Functions to create easier networks ###

def assertAttr(net: BaseNet, target: str, new: list):
    if targetIn:= getattr(net, target, False):
        targetIn += new
    else:
        setattr(net, target, new)

def assertNotAttr(net: BaseNet, target: str):
    assert not hasattr(net, target), 'BaseNet has already a %s layer declared!'%target

def putLinear(net:BaseNet, inputs:int, outputs:int, i:int):
    target  = 'fc%d' % i
    assertNotAttr(net, target)
    setattr(net, target, nn.Linear(inputs, outputs))
    return getattr(net, target)

def construcLinear(net:BaseNet, inputs:int, outputs:int, *hiddenLayers, offset: int = 0):
    first, i, created = inputs, 1 + offset, []
    for layer in hiddenLayers:
        created.append(putLinear(net, first, layer, i))
        i += 1
        first = layer
    created.append(putLinear(net, first, outputs, i))
    assertAttr(net, '_layers_', created)
    return created

def putConv(net:BaseNet, channelIn: int, channelOut: int, i: int, kernel,
                stride: int, padding: int, dilation: int):
    target  = 'cv%d' % i
    assertNotAttr(net, target)
    setattr(net, target, nn.Conv2d(channelIn, channelOut, kernel, stride, padding, dilation))
    return getattr(net, target)

def construcConv(net:BaseNet, shapeInput:Tuple[int, int], 
                    channelIn:int, *convDims, offset: int = 0) -> int:
    first, i, created = channelIn, 1 + offset, []
    shapeOut = shapeInput
    for layerTup in convDims:
        lenTup = len(layerTup)
        stride, padding, dilation = 1, 0, 1
        outs, kernel = layerTup[0], layerTup[1]
        if lenTup > 2:
            stride = layerTup[2]
        if lenTup > 3:
            padding = layerTup[3]
        if lenTup > 4:
            dilation = layerTup[4]
        created.append(putConv(net, first, outs, i, kernel, stride, padding, dilation))
        i += 1
        kernelTuple = isinstance(kernel, tuple)
        kH = kernel if not kernelTuple else kernel[0]
        kW = kernel if not kernelTuple else kernel[1]
        shapeOut = (sqrConvDim(shapeOut[0], kH, stride, padding, dilation),\
            sqrConvDim(shapeOut[0], kW, stride, padding, dilation))
        first = outs
    assertAttr(net, '_convLayers_', created)
    return multiplyIter(shapeOut) * outs, created

def forwardLinear(net: BaseNet, x, offsetBeg: int = 0, offsetEnd: int = 0) -> TENSOR:
    '''
        net should be constructed by construcLinear and
        any non linearity should be declared as noLinear
        This in cpu time difference is little, on gpu 
        this call is more expensive than hardcoding.
    '''
    layers, noLinear = net._layers_, net.noLinear
    for n in range(offsetBeg, len(layers) - 1 - offsetEnd):
        x = noLinear(layers[n](x))
    x = layers[-1 - offsetEnd](x)
    return x

def forwardConv(net: BaseNet, x, offsetBeg: int = 0, offsetEnd: int = 0) -> TENSOR:
    layers, noLinear = net._convLayers_, net.noLinear
    for n in range(offsetBeg, len(layers) - offsetEnd):
        x = noLinear(layers[n](x))
    return x

def layersFromConfig(config: dict, key: str = 'network'):
    """
        Using x for any positive integer.

        Expecting configurations as following:
        - linear layers: 'linear_x' -> int,
            size of the nodes for that hidden fully connected layer. 
        - convolutional 2d layers: 'conv2d_layer_x' -> 
        (channels_out, kernel, stride, padding, dilation)
        -
    """
    import re
    netConfig = config['policy'][key]
    hiddenFC, convLayers = [], []
    # just for linear and convs
    for k in netConfig.keys():
        if re.fullmatch('linear_\d+', k):
            hiddenFC.append((int(k[7:]), netConfig[k], k))
        elif re.fullmatch('conv2d_\d+', k):
            layerConfig = netConfig[k]
            lC = len(layerConfig)
            if len(layerConfig) < 2:
                raise ValueError('%s needs to have at least channels and kernel'%k)
            convLayers.append((int(k[7:]), layerConfig, k))
        # other cases in here
    assertLayers(hiddenFC)
    assertLayers(convLayers)
    layers = {'linear' : hiddenFC,
                'conv2d': convLayers,
                '': []}
    return layers

def assertLayers(layers:list):
    # inplace operation
    if layers == []:
        return
    layers.sort(key = lambda t : t[0])
    prev = 'none'
    for n, tup in enumerate(layers):
        if n + 1 != tup[0]:
            raise ValueError('The layer %s is probably skipping a value.\
                \nPrevious layer is %s. Check the sequence of layers.'%(tup[2], prev))
        prev = tup[2]
        layers[n] = tup[1]

def simpleActionProc(action, discrete: bool):
    if isBatch(action):
        return simpleActionProcBatch(action, discrete)
    if discrete and isItem(action):
        action = action.item()
    else:
        action = action.cpu().squeeze(0).numpy()
    return action

def simpleActionProcBatch(action, discrete: bool):
    actions = []
    if discrete and isItem(action[0]):
        for a in action:
            actions.append(a.cpu().item())
    else:
        actions = action.cpu().numpy()
    return actions

def simpleActionUnProc(action, device):

    if isinstance(action, TENSOR):
        pass
    elif isinstance(action, ARRAY):
        action = torch.from_numpy(action)
    elif isinstance(action, (int, float, list)):
        action = torch.tensor(action)
    else:
        raise NotImplementedError

    return action.to(device)
