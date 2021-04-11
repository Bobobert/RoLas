from rofl.functions.const import *
from .base import QValue

class forestFireDRQNlstm(QValue):
    name = "ff_drqn_lstm"
    def __init__(self, config):
        super(forestFireDRQNlstm, self).__init__()
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config
        
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        lHist = 1
        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.ru1 = nn.LSTM(lHist * 12 * dim**2 + 2, 328,
                            batch_first=True)
        self.fc1 = nn.Linear(328, actions) # from V1
    
    def forward(self, obs, hidden):
        frame, pos= obs["frame"], obs["position"]
        # For RNN handling single channel only
        x = self.rectifier(self.cv1(frame.unsqueeze(1))) 
        x = self.rectifier(self.cv2(x))
        # Features extraction
        x = Tcat([x.flatten(1), pos], dim=1)
        # Recurrent forward
        x , h0 = self.ru1(x.unsqueeze(1), hidden[0])
        hidden[0] = h0
        return self.fc1(x).squeeze(1)

    def new(self):
        new = forestFireDRQNlstm(self.config)
        return new

class forestFireDRQNgru(QValue):
    name = "ff_drqn_gru"
    def __init__(self, config):
        super(forestFireDRQNgru, self).__init__()
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config= config
        
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        lHist = 1
        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.ru1 = nn.GRU(lHist * 12 * dim**2 + 2, 328,
                            batch_first=True)
        self.fc1 = nn.Linear(328, actions) # from V1
    
    def forward(self, obs, hidden):
        frame, pos= obs["frame"], obs["position"]
        # For RNN handling single channel only
        x = self.rectifier(self.cv1(frame.unsqueeze(1))) 
        x = self.rectifier(self.cv2(x))
        # Features extraction
        x = Tcat([x.flatten(1), pos], dim=1)
        # Recurrent forward
        x , h0 = self.ru1(x.unsqueeze(1), hidden[0])
        hidden[0] = h0
        return self.fc1(x).squeeze(1)

    def new(self):
        new = forestFireDRQNgru(self.config)
        return new