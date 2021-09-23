from rofl.functions.const import *
from rofl.functions.functions import np, nb, nprnd
from rofl.functions.torch import *

BASELINE_CONFIG_DEFT = {"optimizer": OPTIMIZER_DEF,
                        "learning_rate":OPTIMIZER_LR_DEF,
                        "minibatch_size" : MINIBATCH_SIZE,
                        "batch_minibatches" : 10,
                        }

@nb.njit
def calculateReturns(rewards: List, notTerminals: List, gamma:float, lmbd:float):
    n = len(rewards)
    newArr = np.zeros(n, dtype = np.float32)
    gae = gamma * lmbd
    newArr[-1] = rewards[-1]
    for i in range(n - 2, -1, -1):
        newArr[i] = rewards[i] + gae * newArr[i + 1] * notTerminals[i]
    return newArr

def unpackBatch(*trajectories, device):
    key = trajectories[0].get("obsType", "standard")
    if key == "framePos":
        return unpackBatchComplexObs(*trajectories, device = device)
    # Ingest the trajectories
    if len(trajectories) > 1:
        states, actions, returns,logprobs, adv  = [], [], [], [], []
        for trajectory in trajectories:
            states += [trajectory["state"]]
            actions += [trajectory["action"]]
            returns += [trajectory["return"]]
            logprobs += [trajectory["prob"]]
            adv += [trajectory["advantage"]]
        states = Tcat(states, dim=0)
        actions = Tcat(actions, dim=0)
        returns = Tcat(returns, dim=0)
        logprobs = Tcat(logprobs, dim=0)
        adv = Tcat(adv) if adv[0] != None else None
    else:
        trajectoryBatch = trajectories[0]
        states = trajectoryBatch["state"]
        actions = trajectoryBatch["action"]
        returns = trajectoryBatch["return"]
        logprobs = trajectoryBatch["prob"]
        adv = trajectoryBatch["advantage"]

    states = states.to(device).requires_grad_()
    actions = actions.to(device)
    returns = returns.to(device)
    logprobs = logprobs.to(device)
    adv = adv.to(device) if isinstance(adv, TENSOR) else None
    return states, actions, returns, adv, logprobs

def unpackBatchComplexObs(*trajectories, device):
    # Ingest the trajectories
    if len(trajectories) > 1:
        states, actions, returns,logprobs, adv  = [], [], [], [], []
        pos, tms = [], []
        for trajectory in trajectories:
            states += [trajectory["state"]["frame"]]
            pos += [trajectory["state"]["position"]]
            tms += [trajectory["state"]["time"]]
            actions += [trajectory["action"]]
            returns += [trajectory["return"]]
            logprobs += [trajectory["prob"]]
            adv += [trajectory["advantage"]]
        states = Tcat(states)
        pos = Tcat(pos)
        tms = Tcat(tms)
        actions = Tcat(actions)
        returns = Tcat(returns)
        logprobs = Tcat(logprobs)
        adv = Tcat(adv) if adv[0] != None else None

    else:
        trajectoryBatch = trajectories[0]
        states = trajectoryBatch["state"]["frame"]
        pos = trajectoryBatch["state"]["position"]
        tms = trajectoryBatch["state"]["time"]
        actions = trajectoryBatch["action"]
        returns = trajectoryBatch["return"]
        logprobs = trajectoryBatch["prob"]
        adv = trajectoryBatch["advantage"]

    states = states.to(device).requires_grad_()
    pos = pos.to(device).float().requires_grad_()
    tms = tms.to(device).float().requires_grad_()
    states = {"frame":states, "position":pos, "time":tms}
    actions = actions.to(device)
    returns = returns.to(device)
    logprobs = logprobs.to(device)
    adv = adv.to(device) if isinstance(adv, TENSOR) else None
    return states, actions, returns, adv, logprobs

class Memory():
    def __init__(self, config):
        self.gamma = config["agent"]["gamma"]
        self.lmbd = config["agent"]["lambda"]
        self.gae = config["agent"]["gae"]
        self.config = config
        self.empties()

    def empties(self):
        self.states = []
        self.actions, self.probs = [], []
        self.rewards = List()
        self.notTerminals = List()
        self.advantages = List()
        self._i = 0

    def clean(self):
        self.empties()

    def __len__(self):
        return self._i

    def add(self, state, action, prob, reward, terminal, advantage = None):
        if self.gae and advantage == None:
            raise ValueError("When GAE is set the advantage value must be passed as a tensor")
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.advantages.append(advantage)
        self.notTerminals.append(0.0 if terminal else 1.0)
        self._i += 1

    def sample(self, size: int, device = DEVICE_DEFT):
        """
        Once available the number of size of experience tuples on the memory
        a batch can be made from the samples.
        """
        if size > len(self):
            return None

        returns = array2Tensor(calculateReturns(self.rewards,self.notTerminals, self.gamma, 1.0), 
                        device = device, grad = False).squeeze()

        states = Tcat(self.states).to(device)
        actions = Tcat(self.actions).to(device)
        probs = Tcat(self.probs).to(device)

        if self.gae:
            advantages = array2Tensor(calculateReturns(self.advantages, self.notTerminals, self.gamma, self.lmbd), 
                                device=device, grad=False).squeeze()
        else:
            advantages = None

        if size < 0:
            sample = {
                "state": states,
                "action": actions,
                "prob": probs,
                "return": returns,
                "advatage": advantages,
            }
        else:
            batchIdx = torch.randperm(len(self))[:size]
            sample = {
                "state":states[batchIdx],
                "action":actions[batchIdx],
                "prob":probs[batchIdx],
                "return":returns[batchIdx],
                "advatage":advantages[batchIdx] if self.gae else None}

        return sample

class MemoryFF(Memory):
    def __init__(self, config):
        super().__init__(config)
        nRow, nCol = config["env"]["n_row"], config["env"]["n_col"]
        if config["agent"].get("scale_pos", False):
            nRow, nCol = 1,1
        self.nRow, self.nCol = nRow, nCol

    def empties(self):
        super().empties()
        self.pos = []
        self.tms = []

    def add(self, state, action, prob, reward, terminal, advantage = None):
        super().add(state["frame"], action, prob, reward, terminal, advantage)
        pos = state["position"]
        pos[0][0] = pos[0][0] / self.nRow
        pos[0][1] = pos[0][1] / self.nCol
        self.pos.append(pos)
        self.tms.append(state.get("time", torch.zeros([1,])))

    def sample(self, size: int, device = DEVICE_DEFT):
        """
        Once available the number of size of experience tuples on the memory
        a batch can be made from the samples.
        """
        if size > len(self):
            return None

        returns = array2Tensor(calculateReturns(self.rewards,self.notTerminals, self.gamma, 1.0), 
                        device = device, grad = False).squeeze()


        frames = Tcat(self.states).to(device)
        pos = Tcat(self.pos).to(device)
        tms = Tcat(self.tms).to(device)
        actions = Tcat(self.actions).to(device)
        probs = Tcat(self.probs).to(device)

        if self.gae:
            advantages = array2Tensor(calculateReturns(self.advantages, self.notTerminals, self.gamma, self.lmbd), 
                                device=device, grad=False).squeeze()
        else:
            advantages = None

        if size < 0:
            sample = {
                "state": {"frame":frames, "position":pos, "time":tms},
                "action": actions,
                "prob": probs,
                "return": returns,
                "advantage": advantages,
            }
        else:
            batchIdx = nprnd.randint(len(self), size = size)
            sample = {
                "state":{"frame":frames[batchIdx], "position":pos[batchIdx], "time":tms[batchIdx]},
                "action":actions[batchIdx],
                "prob":probs[batchIdx],
                "return":returns[batchIdx],
                "advantage":advantages[batchIdx] if self.gae else None}

        sample["obsType"] = "framePos"
        return sample

