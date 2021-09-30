from .base import Actor, Value, ActorCritic, construcConv, construcLinear,\
    forwardConv, forwardLinear, layersFromConfig
from rofl.functions.const import *
from rofl.functions.functions import nn, F, sqrConvDim, Tcat, inputFromGymSpace
from rofl.functions.distributions import Categorical

class gymActor(Actor):
    name = "simple gym actor"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.discrete = True
        self.noLinear = F.relu

        inputs = inputFromGymSpace(config)
        outs = config["policy"]["n_actions"]
        hidden = layersFromConfig(config)
        construcLinear(self, inputs, outs, *hidden['linear'])

    def forward(self, obs):
        return forwardLinear(self, obs)

    def getDist(self, params):
        return Categorical(logits = params)

    def new(self):
        return gymActor(self.config)

class gymActorOld(Actor):
    # in favor in not hardcodding everything... this is now old
    name = "simple gym actor"

    def __init__(self, config):
        super().__init__()
        self.config = config

        inputs = inputFromGymSpace(config)
        outs = config["policy"]["n_actions"]
        h0 = config['policy']['network'].get('size_hidden_1', 30)
        '''h1 = config['policy']['network']['net_hidden_2']
        h2 = config['policy']['network']['net_hidden_3']
        h3 = config['policy']['network']['net_hidden_4']'''

        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, h0)
        '''self.fc2 = nn.Linear(h0, h1)
        self.fc3 = nn.Linear(h1, h2)
        self.fc4 = nn.Linear(h2, h3)'''
        self.fc5 = nn.Linear(h0, outs)

    def forward(self, obs):
        noLinear = self.rectifier
        x = noLinear(self.fc1(obs))
        '''x = noLinear(self.fc2(x))
        x = noLinear(self.fc3(x))
        x = noLinear(self.fc4(x))'''
        return self.fc5(x)

    def getDist(self, params):
        return Categorical(logits = params)

    def new(self):
        return gymActorOld(self.config)

class gymBaseline(Value):
    name = "simple baseline"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.noLinear = F.relu

        inputs = inputFromGymSpace(config)
        hiddens = layersFromConfig(config, 'baseline')
        construcLinear(self, inputs, 1, *hiddens['linear'])

    def forward(self, obs):
        return forwardLinear(self, obs)

    def new(self):
        return gymBaseline(self.config)

class gymAC(ActorCritic, gymActor):
    name = 'simple gym actor critic'
    def __init__(self, config):
        super().__init__(config)
        



class forestFireActorPG(Actor):
    name = "forestFire_pg_actor"
    discrete = True
    def __init__(self, config):
        # Same as forestFireDQNv2, but son of Actor, not qValue
        super(forestFireActorPG, self).__init__()
        self.config = config

        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]

        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2 + 2, 328)
        self.fc2 = nn.Linear(328, actions) # from V1
    
    def forward(self, obs):
        frame, pos = obs["frame"], obs["position"]
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = Tcat([x.flatten(1), pos], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def new(self):
        return forestFireActorPG(self.config)

    def getDist(self, params):
        return Categorical(logits = params)

class ffActor(Actor):

    name = "ff actor"

    def __init__(self, config):
        super(ffActor, self).__init__()
        self.config = config
        self.discrete = True
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]

        self.frameShape = (lHist, *obsShape)
        self.outputs = actions
        self.noLinear = F.relu

        layers = layersFromConfig(config)
        features = construcConv(self, obsShape, lHist, *layers['conv2d'])
        construcLinear(self, features, actions, *layers['linear'])
    
    def forward(self, obs):
        frame = obs[:,:-1].reshape(-1, *self.frameShape)
        x = forwardConv(self, frame)
        x = Tcat([x, obs[:,-1]], dim = 1)
        return forwardLinear(self, x)
        
    def getDist(self, params):
        return Categorical(logits = params)

    def new(self):
        return ffActor(self.config)

class ffActorOld(Actor):
    h0 = 328
    name = "ff_actor_channel"
    def __init__(self, config):
        super(ffActor, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config.get("net_hidden_1", self.h0)
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist * 3,  18, 8, 4, padding=1)
        dim = sqrConvDim(obsShape[0], 8, 4, 1)
        self.cv2 = nn.Conv2d( 18,  36, 4, 2, padding=1)
        dim = sqrConvDim(dim, 4, 2, 1)
        self.cv3 = nn.Conv2d( 36,  36, 3, 1, padding=1)
        dim = sqrConvDim(dim, 3, 1, 1)
        self.fc1 = nn.Linear(lHist * 36 * dim**2 + 1, h0)
        self.fc2 = nn.Linear(h0, actions)
    
    def forward(self, obs):
        frame, pos, t = obs["frame"], obs["position"], obs["time"]
        frame = frame.reshape(frame.shape[0], frame.shape[1] * frame.shape[4], frame.shape[2], frame.shape[3])
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.cv3(x))
        x = Tcat([x.flatten(1), t], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def getDist(self, params):
        return Categorical(logits = params)

    def new(self):
        return ffActor(self.config)

class ffBaseline(Value):
    h0 = 328
    name = "ff_baseline_channel"
    def __init__(self, config):
        super(ffBaseline, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config.get("net_hidden_1", self.h0)
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist * 3,  18, 8, 4, padding=1)
        dim = sqrConvDim(obsShape[0], 8, 4, 1)
        self.cv2 = nn.Conv2d( 18,  36, 4, 2, padding=1)
        dim = sqrConvDim(dim, 4, 2, 1)
        self.cv3 = nn.Conv2d( 36,  36, 3, 1, padding=1)
        dim = sqrConvDim(dim, 3, 1, 1)
        self.fc1 = nn.Linear(lHist * 36 * dim**2 + 1, h0)
        self.fc2 = nn.Linear(h0, 1)
    
    def forward(self, obs):
        frame, pos, t = obs["frame"], obs["position"], obs["time"]
        frame = frame.reshape(frame.shape[0], frame.shape[1] * frame.shape[4], frame.shape[2], frame.shape[3])
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.cv3(x))
        x = Tcat([x.flatten(1), t], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def new(self):
        return ffBaseline(self.config)

class forestFireBaseline(Value):
    name = "forestFire_baseline"
    def __init__(self, config):
        # Same as forestFireDQNv2, but son of Actor, not qValue
        super(forestFireBaseline, self).__init__()
        self.config = config

        lHist = config["agent"]["lhist"]
        obsShape = config["env"]["obs_shape"]

        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2 + 2, 328)
        self.fc2 = nn.Linear(328, 1) # from V1
    
    def forward(self, obs):
        frame, pos = obs["frame"], obs["position"]
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = Tcat([x.flatten(1), pos], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

    def new(self):
        return forestFireBaseline(self.config)
