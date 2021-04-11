from rofl.functions.const import *
from rofl.functions.torch import *
from rofl.policies.dqn import dqnPolicy
from rofl.utils.drqn import recurrentArguments, unpackBatch

# TODO COmplete update for this policy
def drqnTarget(onlineNet, targetNet, s1, s2, r, t, gamma, boot, hidden, 
                double:bool = True):
    with no_grad():
        # First use the hidden from onlineNet
        if double:
            On_model_out = onlineNet.forward(s2, hidden)
            a_greedy = On_model_out.max(1)[1]
        # Boot the target network
        hidden.reset()
        bootRecurrent(targetNet, boot, hidden)
        _ = targetNet.forward(s1, hidden)
        model_out = targetNet.forward(s2, hidden)
        if double:
            Qs2_max = model_out.gather(1, a_greedy.unsqueeze(1)).squeeze(1)
        else:
            Qs2_max = model_out.max(1)[0] 
        target = r + Tmul(t, Qs2_max).mul(gamma).reshape(r.shape)
    return target.unsqueeze(1)

def bootRecurrent(net, bootStates, hidden):
    r = len(bootStates["frame"])
    with no_grad():
        for i in range(r-1, -1, -1):
            bootState = {"frame":bootStates["frame"][i],"position":bootStates["position"][i]}
            net.forward(bootState, hidden)

class drqnPolicy(dqnPolicy):
    name = "drqnPolicy"

    def __init__(self, config, dqn, tbw = None):
        super(drqnPolicy, self).__init__(config, dqn, tbw)

        self.recurrentBoot = config["policy"].get("recurrent_boot", 10)
        self.recurrentStateBatch = recurrentArguments(config)
        self.recurrentState = recurrentArguments(config)
    
    def getAction(self, state):
        throw = np.random.uniform()
        eps = self.epsilon.test(state) if self.test else self.epsilon.train(state)
        with no_grad(): # Always accumulates the hidden state
            outValue = self.dqnOnline(state, self.recurrentState)
        if throw <= eps:
            return self.getRandom()
        else:
            return outValue.argmax(1).item()

    def update(self, *infoDicts):
        st1, st2, rewards, actions, dones, boot = unpackBatch(*infoDicts, device = self.device)
        # Make zeros the hidden states
        hidden = self.recurrentStateBatch
        hidden.initHidden(device = self.device)
        bootRecurrent(self.dqnOnline, boot, hidden)
        qValues = self.dqnOnline(st1, hidden).gather(1, actions)
        qTargets = drqnTarget(self.dqnOnline, self.dqnTarget, 
                                st1, st2, rewards, dones, self.gamma, 
                                boot, hidden, self.double)

        loss = F.smooth_l1_loss(qValues, qTargets, reduction="mean")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tbw is not None:
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            self.tbw.add_scalar('train/Mean TD Error', torch.mean(qTargets - qValues).item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.eva_meang, self.eva_maxg)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1
    
    def resetHidden(self):
        self.recurrentState.initHidden(batch=1, device = self.device)