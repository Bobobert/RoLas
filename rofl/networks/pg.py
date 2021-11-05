from .base import Actor, Value, ActorCritic, construcConv, construcLinear,\
    forwardConv, forwardLinear, layersFromConfig
from rofl.functions.const import ARRAY
from rofl.functions.functions import Texp, Tcat, F, multiplyIter, outputFromGymSpace, inputFromGymSpace
from rofl.functions.distributions import Categorical, Normal
from rofl.utils.bulldozer import composeMultiDiscrete, decomposeMultiDiscrete, decomposeObsWContextv0

class GymActor(Actor):
    name = 'simple gym actor'

    def __init__(self, config):
        super().__init__(config)
        continuos = config['policy']['continuos']
        self.discrete = not continuos
        self.noLinear = F.relu

        inputs = inputFromGymSpace(config)
        outs = config['policy']['n_actions']
        outs = outputFromGymSpace(config) if outs is None else outs
        outs *= 2 if continuos else 1
        hidden = layersFromConfig(config)
        construcLinear(self, inputs, outs, *hidden['linear'])

    def forward(self, obs):
        return forwardLinear(self, obs)

    def getDist(self, params):
        if self.discrete:
            return Categorical(logits=params)
        else:
            n = params.shape[-1] // 2
            return Normal(params[:,:n], Texp(params[:,n:]))

class GymBaseline(Value):
    name = 'simple baseline'

    def __init__(self, config):
        super().__init__(config)
        self.noLinear = F.relu

        inputs = inputFromGymSpace(config)
        hiddens = layersFromConfig(config, 'baseline')
        construcLinear(self, inputs, 1, *hiddens['linear'])

    def forward(self, obs):
        return forwardLinear(self, obs)

class GymAC(ActorCritic):
    name = 'simple gym actor critic'
    def __init__(self, config):
        super().__init__(config)
        continuos = config['policy']['continuos']
        self.discrete = not continuos
        self.noLinear = F.relu

        inputs = inputFromGymSpace(config)
        outs = config['policy']['n_actions']
        outs = outputFromGymSpace(config) if outs is None else outs
        outs *= 2 if continuos else 1
        hidden = layersFromConfig(config)['linear']
        lastHidden, self._lLayers = hidden[-1], len(hidden)
        construcLinear(self, inputs, lastHidden, *hidden[:-1])
        [self.actLayer] = construcLinear(self, lastHidden, outs, offset=len(hidden))
        [self.valLayer] = construcLinear(self, lastHidden, 1, offset=len(hidden) + 1)

    def sharedForward(self, x):
        return self.noLinear(forwardLinear(self, x, offsetEnd=2))

    def valueForward(self, x):
        return self.valLayer(x)

    def actorForward(self, x):
        return self.actLayer(x)

    def getDist(self, params):
        if self.discrete:
            return Categorical(logits=params)
        else:
            n = params.shape[-1] // 2
            return Normal(params[:,:n], Texp(params[:,n:]))

class FFActorCritic(ActorCritic):
    name = 'ff actor'
    def __init__(self, config):
        super().__init__(config)
        self.discrete = True
        self.noLinear = F.relu

        self.actionSpace = config['env']['action_space']
        actions = config['policy']['n_actions']
        if isinstance(actions, (tuple, ARRAY)):
            actions = multiplyIter(actions)
        lHist = config['agent']['lhist']
        channels = config['agent'].get('channels', 1)
        obsShape = config['env']['obs_shape']
        self.frameShape = (lHist * channels, *obsShape)

        layers = layersFromConfig(config)
        features, _ = construcConv(self, obsShape, lHist * channels, *layers['conv2d'])
        
        linearLayers = layers['linear']
        self.actorLayers = construcLinear(self, features + 3, actions, *linearLayers)
        self.offset = len(linearLayers) + 1
        self.valueLayers = construcLinear(self, features + 3, 1, *linearLayers, offset=self.offset)

    def sharedForward(self, observation):
        frame, context = decomposeObsWContextv0(observation, self.frameShape)
        x = forwardConv(self, frame)
        x = self.noLinear(x)
        return Tcat([x.flatten(1), context], dim=1)

    def actorForward(self, observation):
        return forwardLinear(self, observation, offsetEnd=self.offset)

    def valueForward(self, observation):
        return forwardLinear(self, observation, self.offset)

    def processAction(self, action):
        return composeMultiDiscrete(action, self.actionSpace)

    def unprocessAction(self, action, batch: bool):
        return decomposeMultiDiscrete(action, self.actionSpace, batch, self.device).squeeze_()

    def getDist(self, params):
        return Categorical(logits=params)
