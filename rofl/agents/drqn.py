from .dqn import dqnAtariAgent
from rofl.utils.dqn import *
from rofl.utils.drqn import recurrentArguments, MemoryReplayRecurrentFF

class drqnFFAgent(dqnAtariAgent):
    name = "drqnForestFireAgentv0"

    def __init__(self, config, policy, envMaker,
                    tbw = None, useTQDM = False):
        # From the original agent
        super(drqnFFAgent, self).__init__(config, policy, envMaker,
                                            tbw = tbw, useTQDM=useTQDM)

        obsShape = config["env"]["obs_shape"]
        self.memory = MemoryReplayRecurrentFF(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        recurrent_boot = config["policy"]["recurrent_boot"])
        self.lastPos = None

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        frame, pos = obs["frame"], obs["position"]
        if reset:
            self.policy.resetHidden()
        procsFrame = imgResize(frame, size = self.frameSize)
        self.lastFrame = {"frame":procsFrame, "position":pos}
        newObs = torch.from_numpy(procsFrame).to(self.device).unsqueeze(0).float().div(255)
        Tpos = torch.as_tensor(pos).to(self.device).float().unsqueeze(0)
        return {"frame": newObs, "position":Tpos}
    
    def reportCustomMetric(self):
        return 0.0