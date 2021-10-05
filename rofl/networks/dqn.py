from rofl.functions.functions import *
from .base import QValue, construcConv, construcLinear, forwardConv, forwardLinear, layersFromConfig

class dqnAtari(QValue):
    name = 'deep Qnetwork atari'

    def __init__(self, config):
        super().__init__(config)
        self.noLinear = F.relu

        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.frameShape = (lHist, *obsShape)
        self.outputs = actions
        
        self.configLayers = layers = layersFromConfig(config)
        self.features, _ = construcConv(self, obsShape, lHist, *layers['conv2d'])
        construcLinear(self, self.features, actions, *layers['linear'])

    def forward(self, obs):
        x = forwardConv(self, obs)
        x = x.flatten(1)
        return forwardLinear(self, x)

class dqnAtariDueling(dqnAtari):
    def __init__(self, config):
        super().__init__(config)
        actions = config["policy"]["n_actions"]
        linearLayers = self.configLayers['linear']
        self.linearOffset = offset = len(linearLayers) + 1
        construcLinear(self, self.features, actions, *linearLayers, offset = offset)

    def forward(self, obs):
        x = forwardConv(self, obs)
        x = x.flatten(1)
        xVal = x.clone()
        offset = self.linearOffset
        xA = forwardLinear(self, x, offsetEnd = offset)
        xVal = forwardLinear(self, x, offsetBeg = offset)

        Amean = Tmean(xA, dim=1, keepdim=True)
        return xVal + (xA - Amean)

## TO BE DELETED ## TODO
class forestFireDQNVanilla(QValue):
    """
    Policy network for DQN-Atari

    parameters
    ----------
    lHist: int
        Number of frames on the stack for a history. The frame size 
        is defined as (84, 84)
    actions: int
        Number of actions in which the policy will chose

    """
    def __init__(self, config):
        super(forestFireDQNVanilla, self).__init__()
        # Variables
        obsShape = config["env"]["obs_shape"]
        h0 = config["policy"]['network'].get("linear_1", 512)
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.name = 'DQN-policy'
        # Operational
        # Net
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, 32, 8, 4)
        dim = sqrConvDim(obsShape[0], 8, 4)
        self.cv2 = nn.Conv2d(32, 64, 4, 2)
        dim = sqrConvDim(dim, 4, 2)
        self.cv3 = nn.Conv2d(64, 64, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(64 * dim ** 2, h0)
        self.fc2 = nn.Linear(h0, actions) # from fully connected to actions

    def forward(self, X):
        X = self.cv1(X)
        X = self.rectifier(X)
        X = self.cv2(X)
        X = self.rectifier(X)
        X = self.cv3(X)
        X = self.rectifier(X)
        X = self.fc1(X.flatten(1))
        X = self.rectifier(X)
        return self.fc2(X)

class forestFireDQN(QValue):
    def __init__(self, config):
        super(forestFireDQN, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config["policy"]['network'].get("linear_1", 328)
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2, h0)
        self.fc2 = nn.Linear(h0, actions) # V1
    
    def forward(self, x):
        x = self.rectifier(self.cv1(x))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.fc1(x.flatten(1)))
        return self.fc2(x)

class forestFireDQNv2(QValue):
    def __init__(self, config):
        super(forestFireDQNv2, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config["policy"]['network'].get("linear_1", 328)
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2 + 2, h0)
        self.fc2 = nn.Linear(h0, actions) # from V1
    
    def forward(self, obs):
        frame, pos = obs["frame"], obs["position"]
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = Tcat([x.flatten(1), pos], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)

class forestFireDQNv3(QValue):
    def __init__(self, config):
        super(forestFireDQNv3, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config["policy"]['network'].get("linear_1", 328)
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
        self.fc1 = nn.Linear( 36 * dim**2 + 1, h0)
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

class forestFireDQNres(QValue):

    def __init__(self, config):
        super(forestFireDQNres, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape[:2]
        self.channels = chls = obsShape[2]
        self.rectifier = F.relu
        nCh = lHist * chls
        self.cv1 = nn.Conv2d(nCh, 256, 3, 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        dim = sqrConvDim(obsShape[0], 3, 1)
        self.cv2 = nn.Conv2d(256, 256, 3, 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(256)
        dim = sqrConvDim(dim, 3, 1)
        self.cv3 = nn.Conv2d(256, 256, 3, 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.cv4 = nn.Conv2d(256, 2, 1, 1)
        self.bn4 = nn.BatchNorm2d(2)
        dim = sqrConvDim(dim, 1, 1)
        self.fc1 = nn.Linear(2 * obsShape[0] * obsShape[1], actions)

    def forward(self, obs):
        # TODO
        obs = obs.reshape(obs.shape[0], obs.shape[1] * obs.shape[4], obs.shape[2], obs.shape[3])
        r = self.rectifier
        x0 = r(self.bn1(self.cv1(obs)))
        #residual block
        x = r(self.bn2(self.cv2(x0)))
        x = r(self.bn3(self.cv3(x)) + x0)
        # output
        x = r(self.bn4(self.cv4(x)))
        return self.fc1(x.flatten(1))

class forestFireDuelingDQN(QValue):
    def __init__(self, config):
        super(forestFireDuelingDQN, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        h0 = config["policy"]['network'].get("linear_1", 328)
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2 + 2, h0)
        self.fc2 = nn.Linear(h0, actions) # from V1
        self.fc3 = nn.Linear(lHist * 12 * dim**2 + 2, h0)
        self.fc4 = nn.Linear(h0, 1)
    
    def forward(self, obs):
        frame, pos = obs["frame"], obs["position"]
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        features = Tcat([x.flatten(1), pos], dim=1)
        xV = features.clone()
        xA = features.clone()
        xA = self.rectifier(self.fc1(xA))
        xA = self.fc2(xA)
        xV = self.rectifier(self.fc3(xV))
        xV = self.fc4(xV)
        Amean = Tmean(xA, dim=1, keepdim=True)

        return xV + (xA - Amean)

    def new(self):
        new = forestFireDuelingDQN(self.config)
        return new

class forestFireDQNth0(QValue):
    def __init__(self, config):
        super(forestFireDQNth0, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config
        config = config["policy"]['network']
        h0 = config.get("net_hidden_1", 256)
        h1 =config.get("net_hidden_2", 512)
        h2 = config.get("net_hidden_3", 256)
        h3 = config.get("net_hidden_4", 64)
        
        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu
        ins = obsShape[0] * obsShape[1] * obsShape[2]
        self.fc1 = nn.Linear(ins + 3, h0)
        self.fc2 = nn.Linear(h0, h1)
        self.fc3 = nn.Linear(h1, h2)
        self.fc4 = nn.Linear(h2, h3)
        self.fc5 = nn.Linear(h3, actions) # V1
    
    def forward(self, obs):
        frame, pos, t = obs["frame"], obs["position"], obs["time"]
        frame = frame.flatten(1)
        obs = Tcat([frame, pos, t], dim = 1)
        rec = self.rectifier
        
        X = rec(self.fc1(obs))
        X = rec(self.fc2(X))
        X = rec(self.fc3(X))
        X = rec(self.fc4(X))

        return self.fc5(X)

    def new(self):
        new = forestFireDQNth0(self.config)
        return new

def processObs(obj, obs, channels = False):
    frame, pos_t = obs[:,:-3], obs[:,-3:]
    if channels:
        frame = frame.reshape((-1, obj.lHist * obj.obsShape[-1], *obj.obsShape[:-1]))
    else:
        frame = frame.reshape((-1, obj.lHist, obj.obsShape))
    return frame, pos_t

class ffDQNv4(forestFireDQN):
    name = "forestFire DQN v4"

    def forward(self, obs):
        frame, pos_t = processObs(self, obs)
        return super().forward(frame)

    def new(self):
        return ffDQNv4(self.config)

class ffDQNv5(forestFireDQNv3):
    name = "forestFire DQN v5"

    def forward(self, obs):
        frame, pos_t = processObs(self, obs)
        x = self.rectifier(self.cv1(frame))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.cv3(x))
        x = Tcat([x.flatten(1), pos_t[:,-1]], dim=1)
        x = self.rectifier(self.fc1(x))
        return self.fc2(x)        

    def new(self):
        return ffDQNv5(self.config)
