from rofl.functions.const import *
from rofl.functions.torch import *
from rofl.policies.dqn import dqnPolicy
from rofl.utils.drqn import recurrentArguments, unpackBatch, newZero

def procState(state):
    if isinstance(state, dict):
        states = []
        frames, positions = state["frame"], state["position"]
        # unpack by batch of the recurrent boot
        for i in range(frames.shape[0]):
            states += [{"frame":frames[i], "position":positions[i]}]
        return states
    return state

def drqnForward(net, states, config, grad = True):
    # Init vars
    c = config["policy"]
    units, size, lstm = c["recurrent_units"], c["recurrent_hidden_size"], c["recurrent_unit"]
    layers = c["recurrent_layers"]
    nBoot, batch = c["recurrent_boot"], c["minibatch_size"]
    h0 = newZero(batch, units * layers, size, net.device, lstm, grad)
    outs, hiddens = [], []
    # Doing forward for every state
    for i in range(nBoot + 1, -1, -1):
        if i == (nBoot + 1):
            out, hidden = net.completeForward(states[i], h0)
        else:
            out, hidden = net.completeForward(states[i], hiddens[-1])
        outs += [out]
        hiddens += [hidden]
    # Returns from the oldest to the newest
    # The last two are s1, s2 from the actual time
    return outs, hidden

def drqnTarget(targetOuts, onlineOuts, rewards, terminals, gamma, double = True):
    targets, n = [], len(targetOuts)
    for i in range(n - 1): # Stop at s1
        # First use the hidden from onlineNet
        if double:
            On_model_out = onlineOuts[i + 1]
            a_greedy = On_model_out.max(1)[1]
        model_out = targetOuts[i + 1]
        if double:
            Qs2_max = model_out.gather(1, a_greedy.unsqueeze(1)).squeeze(1)
        else:
            Qs2_max = model_out.max(1)[0] 
        r, t = rewards[n - 1 - i], terminals[n - 1 - i]
        target = r + Tmul(t, Qs2_max.unsqueeze(1)).mul(gamma).reshape(r.shape)
        targets += [target]
    # From oldest to newest, len(targets) = len(Outs) - 1
    return targets

def drqnGatherActions(values, actions):
    n = len(values)
    gatherValues = []
    for i in range(n - 1): # Stop at s1
        # Actions are from newest to oldest 0 = n
        actions_ = actions[n - 1 - i]
        gatherValues += [values[i].gather(1, actions_)]
    # From oldest to newest, len(targets) = len(Outs) - 1
    return gatherValues

class drqnPolicy(dqnPolicy):
    name = "drqnPolicy"

    def __init__(self, config, dqn, tbw = None):
        super(drqnPolicy, self).__init__(config, dqn, tbw)
        self.clipGrad = config["policy"].get("clip_grad")
        self.recurrentBoot = config["policy"].get("recurrent_boot", 10)
        self.recurrentState = recurrentArguments(config)
        self.zeroHidden = recurrentArguments(config)

    def getAction(self, state):
        throw = nprnd.uniform()
        eps = self.epsilon.test(state) if self.test else self.epsilon.train(state)
        with no_grad(): # Always accumulates the hidden state
            outValue = self.dqnOnline(state, self.recurrentState)
        if throw <= eps:
            return self.getRndAction()
        else:
            return outValue.argmax(1).item()

    def _update_(self, *infoDicts):
        states, rewards, actions, dones, IS = unpackBatch(*infoDicts, device = self.device)
        states = procState(states)
        # Forward Online net
        onlineOut, onlineHiddens = drqnForward(self.dqnOnline, states, self.config, True)
        # Forward Target net
        with no_grad():
            targetOut, targetHiddens = drqnForward(self.dqnTarget, states, self.config, False)
        # Calculate targets and values
        # List from older to newest
        qTargets = drqnTarget(targetOut, onlineOut, rewards, dones, self.gamma, self.double)
        qValues = drqnGatherActions(onlineOut, actions)
        # Calculate and apply losses
        losses = np.zeros(len(qValues))
        self.optimizer.zero_grad()
        for i in range(len(qValues) - 1, -1 , -1):
            loss = F.smooth_l1_loss(qValues[i], qTargets[i], reduction="mean")
            if i == 0:
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            losses[i] = loss.item()
        if self.clipGrad > 0.0:
            clipGrads(self.dqnOnline, self.clipGrad)
        self.optimizer.step()

        if self.tbw != None:
            self.tbw.add_scalar('train/Loss', np.mean(losses), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.evalMeanGrad, self.evalMaxGrad)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1
    
    def update(self, *infoDicts):
        states, rewards, actions, dones = unpackBatch(*infoDicts, device = self.device)
        states = procState(states)
        def makeHidden():
            return self.zeroHidden.initHidden(device = self.device)[0]
        # Forward Online net
        h0 = makeHidden()
        onlineOut, onlineHiddens = self.dqnOnline.seqForward(states, h0)
        # Forward Target net
        with no_grad():
            h0 = makeHidden()
            targetOut, targetHiddens = self.dqnTarget.seqForward(states, h0)
        # Calculate targets and values
        # List from older to newest
        qTargets = Tstack(drqnTarget(targetOut, onlineOut, rewards, dones, self.gamma, self.double))
        qValues = Tstack(drqnGatherActions(onlineOut, actions))
        # Calculate and apply losses
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(qValues, qTargets, reduction="mean")
        loss.backward()
        self.optimizer.step()

        if self.clipGrad > 0.0:
            clipGrads(self.dqnOnline, self.clipGrad)
        self.optimizer.step()

        if self.tbw != None:
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.evalMeanGrad, self.evalMaxGrad)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1
    
    def resetHidden(self):
        self.recurrentState.initHidden(batch=1, device = self.device)
