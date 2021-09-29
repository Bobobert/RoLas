from .base import Actor, Value
from rofl.functions.const import *
from rofl.functions.functions import torch, nn, F, sqrConvDim, Tcat, multiplyIter
from rofl.functions.distributions import Categorical

def inputFromGymSpace(config: dict):
    return multiplyIter(config['env']['observation_space'].shape)

class gymActor(Actor):
    name = "simple gym actor"

    def __init__(self, config):
        super().__init__()
        self.config = config

        inputs = inputFromGymSpace(config)
        outs = config["policy"]["n_actions"]
        h0 = config['policy']['network'].get('net_hidden_1', 30)

        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, h0)
        self.fc2 = nn.Linear(h0, outs)

    def forward(self, obs):
        noLinear = self.rectifier
        x = noLinear(self.fc1(obs))
        return self.fc2(x)

    def getDist(self, params):
        return Categorical(logits = params)

    def new(self):
        return gymActor(self.config)

class gymBaseline(Value):
    name = "simple baseline"

    def __init__(self, config):
        super().__init__()
        self.config = config

        inputs = inputFromGymSpace(config)
        h0 = config['policy']['network'].get('net_hidden_1', 30)

        self.rectifier = F.relu
        self.fc1 = nn.Linear(inputs, h0)
        self.fc2 = nn.Linear(h0, 1)

    def forward(self, obs):
        noLinear = self.rectifier
        x = noLinear(self.fc1(obs))
        return self.fc2(x)

    def new(self):
        return gymBaseline(self.config)

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
