from rofl.functions.functions import F, Tcat, Tmean, nn, sqrConvDim
from rofl.utils.bulldozer import composeMultiDiscrete, decomposeMultiDiscrete, decomposeObsWContextv0
from .base import QValue, construcConv, construcLinear, forwardConv, forwardLinear, layersFromConfig

class DqnAtari(QValue):
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

class DqnAtariDueling(DqnAtari):
    def __init__(self, config):
        super().__init__(config)
        actions = config["policy"]["n_actions"]
        linearLayers = self.configLayers['linear']
        self.linearOffset = offset = len(linearLayers) + 1
        construcLinear(self, self.features, actions, *linearLayers, offset=offset)

    def forward(self, obs):
        x = forwardConv(self, obs)
        x = x.flatten(1)
        xVal = x.clone()
        offset = self.linearOffset
        xA = forwardLinear(self, x, offsetEnd=offset)
        xVal = forwardLinear(self, x, offsetBeg=offset)

        Amean = Tmean(xA, dim=1, keepdim=True)
        return xVal + (xA - Amean)

class DqnCA(QValue):
    name = 'dqn CA w/ support channels'
    def __init__(self, config):
        super().__init__(config)
        self.noLinear = F.relu

        actions = config["policy"]["n_actions"]
        self.actionSpace = config['env']['action_space']
        obsShape = config["env"]["obs_shape"]

        lHist = config["agent"]["lhist"]
        channels = config['agent'].get('channels', 1)
        self.frameShape = (lHist * channels, *obsShape)
        self.outputs = actions
        
        self.configLayers = layers = layersFromConfig(config)
        self.features, _ = construcConv(self, obsShape, lHist * channels, *layers['conv2d'])
        construcLinear(self, self.features + 3, actions, *layers['linear'])

    def forward(self, observation):
        frame, context = decomposeObsWContextv0(observation, self.frameShape)
        x = forwardConv(self, frame)
        x = Tcat([x.flatten(1), context], dim=1)
        return forwardLinear(self, x)
    
    def processAction(self, action):
        return composeMultiDiscrete(action, self.actionSpace)

    def unprocessAction(self, action, batch: bool):
        return decomposeMultiDiscrete(action, self.actionSpace, batch, self.device)

class DqnCADueling(DqnCA):
    name = 'dqn CA dueling w/ support channels'
    def __init__(self, config):
        super().__init__(config)
        actions = config["policy"]["n_actions"]

        linearLayers = self.configLayers['linear']
        self.linearOffset = offset = len(linearLayers) + 1
        construcLinear(self, self.features + 3, actions, *linearLayers, offset=offset)

    def forward(self, observation):
        frame, context = decomposeObsWContextv0(observation, self.frameShape)
        x = forwardConv(self, frame)
        x = Tcat([x.flatten(1), context], dim=1)
        xVal = x.clone()
        offset = self.linearOffset
        xA = forwardLinear(self, x, offsetEnd=offset)
        xVal = forwardLinear(self, x, offsetBeg=offset)

        Amean = Tmean(xA, dim=1, keepdim=True)
        return xVal + (xA - Amean)

## TO BE DELETED ## TODO
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
