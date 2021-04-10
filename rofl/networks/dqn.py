from rofl.functions.const import *
from .base import QValue

class dqnAtari(QValue):
    """
    Policy network for DQN-Atari

    parameters
    ----------
    lHist: int
        Number of frames on the stack for a history. The frame size 
        is defined as (84, 84)
    actions: int
        Number of actions in which the policy will chose
    dropouts: list
        A list with 4 probabilities each to decide for the layers from
        cv1, cv2, cv3 and fc1 if drops some nodes or not.

    """
    def __init__(self, config):
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        self.config = config
        super(dqnAtari, self).__init__()
        # Variables
        self.lHist = lHist
        self.outputs = actions
        self.name = 'DQN-policy'
        # Operational
        # Net
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, 32, 8, 4)
        self.cv2 = nn.Conv2d(32, 64, 4, 2)
        self.cv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, actions) # from fully connected to actions

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

    def new(self):
        new = dqnAtari(self.config)
        return new

class forestFireDQN(QValue):
    def __init__(self, config):
        super(forestFireDQN, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu
        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.fc1 = nn.Linear(lHist * 12 * dim**2, 328)
        self.fc2 = nn.Linear(328, actions) # V1
    
    def forward(self, x):
        x = self.rectifier(self.cv1(x))
        x = self.rectifier(self.cv2(x))
        x = self.rectifier(self.fc1(x.flatten(1)))
        return self.fc2(x)

    def new(self):
        new = forestFireDQN(self.config)
        return new

class forestFireDQNv2(QValue):
    def __init__(self, config):
        super(forestFireDQNv2, self).__init__()
        lHist = config["agent"]["lhist"]
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config

        self.lHist = lHist
        self.outputs = actions
        self.obsShape = obsShape
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
        new = forestFireDQNv2(self.config)
        return new
