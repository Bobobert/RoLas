from rofl.functions.const import *
from .base import QValue

class forestFireDRQNlstm(QValue):
    name = "ff_drqn_lstm"
    h0 = 328
    def __init__(self, config):
        super(forestFireDRQNlstm, self).__init__()
        actions = config["policy"]["n_actions"]
        obsShape = config["env"]["obs_shape"]
        self.config = config
        
        self.outputs = actions
        self.obsShape = obsShape
        self.rectifier = F.relu

        lHist = 1
        self.cv1 = nn.Conv2d(lHist, lHist * 6, 5, 2)
        dim = sqrConvDim(obsShape[0], 5, 2)
        self.cv2 = nn.Conv2d(lHist * 6, lHist * 12, 3, 1)
        dim = sqrConvDim(dim, 3, 1)
        self.inputHiddenSize = lHist * 12 * dim**2 + 2
        outhidden = config["policy"].get("recurrent_hidden_size", self.h0)
        self.ru1 = nn.LSTM(self.inputHiddenSize, outhidden,
                            batch_first=True)
        self.fc1 = nn.Linear(outhidden, actions) # from V1
    
    def cnnForward(self, obs):
        frame, pos= obs["frame"], obs["position"]
        x = self.rectifier(self.cv1(frame.unsqueeze(1))) 
        x = self.rectifier(self.cv2(x))
        return Tcat([x.flatten(1), pos], dim=1).unsqueeze(1)

    def completeForward(self, obs, hiddenT):
        features = self.cnnForward(obs)
        x, h = self.ru1(features, hiddenT)
        x = self.fc1(x).squeeze(1)
        return x, h

    def forward(self, obs, hidden):
        # For RNN handling single channel only
        # Features extraction
        x = self.cnnForward(obs)
        # Recurrent forward
        x , h0 = self.ru1(x, hidden[0])
        hidden[0] = h0
        return self.fc1(x).squeeze(1)

    def new(self):
        new = forestFireDRQNlstm(self.config)
        return new

class forestFireDRQNgru(forestFireDRQNlstm):
    name = "ff_drqn_gru"
    def __init__(self, config):
        super(forestFireDRQNgru, self).__init__(config)
        outhidden = config["policy"].get("recurrent_hidden_size", self.h0)
        self.ru1 = nn.GRU(self.inputHiddenSize, outhidden,
                            batch_first=True)

    def new(self):
        new = forestFireDRQNgru(self.config)
        return new