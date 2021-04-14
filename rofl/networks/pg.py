from .base import Actor, Value
from rofl.functions.const import *
from rofl.functions.distributions import Categorical

class dcontrolActorPG(Actor):
    name = "classic_control_pg_net"
    h0 = 30
    def __init__(self, config):
        super(dcontrolActorPG, self).__init__()
        self.config = config

        inputs = config["policy"]["n_inputs"]
        outs = config["policy"]["n_actions"]
        h0 = self.h0
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
        return dcontrolActorPG(self.config)

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
