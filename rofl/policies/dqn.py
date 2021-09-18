from rofl.functions.const import *
from rofl.utils.dqn import unpackBatch
from rofl.functions.torch import *
from rofl.functions import EpsilonGreedy
from .base import Policy

def dqnTarget(onlineNet, targetNet, s2, r, t, gamma, double:bool = True):
    with no_grad():
        model_out = targetNet.forward(s2)
        if double:
            On_model_out = onlineNet.forward(s2)
            a_greedy = On_model_out.max(1)[1]
            Qs2_max = model_out.gather(1, a_greedy.unsqueeze(1)).squeeze(1)
        else:
            Qs2_max = model_out.max(1)[0] 
        target = r + Tmul(t, Qs2_max).mul(gamma).reshape(r.shape)
    return target.unsqueeze(1)

def importanceNorm(IS):
    IS = IS.unsqueeze(1)
    maxIS = IS.max()
    return IS / maxIS

class dqnPolicy(Policy):
    discrete = True
    name = "dqnPolicy"
    def __init__(self, config, dqn, tbw = None):

        self.config = config.copy()
        self.dqnOnline = self.actor = dqn
        self.dqnTarget = cloneNet(dqn)
        self.epsilon = EpsilonGreedy(config)
        self.gamma = config["agent"]["gamma"]
        self.prioritized = config["agent"]["memory_prioritized"]
        config = config["policy"]
        self.updateTarget = config["freq_update_target"]
        self.nActions = config["n_actions"]
        self.double = config.get("double", False)
        
        parameters, lr = self.dqnOnline.parameters(), config["learning_rate"]
        if config["optimizer"] == "adam":
            self.optimizer = optim.Adam(parameters, lr = lr, **config.get("optimizer_args", {}))
        elif config["optimizer"] == "rmsprop":
            self.optimizer = optim.RMSprop(parameters, lr = lr, **config.get("optimizer_args", {}))
        
        self.epochs = 0
        self.tbw = tbw
        self.eva_maxg = config["evaluate_max_grad"]
        self.eva_meang = config["evaluate_mean_grad"]
        self.lastNetOutput = None
        super(dqnPolicy, self).__init__()

    def getAction(self, state):
        throw = nprnd.uniform()
        eps = self.epsilon.test(state) if self.test else self.epsilon.train(state)
        output = None

        def calOut():
            nonlocal output
            with no_grad():
                output = self.dqnOnline(state)
            self.lastNetOutput = output

        if throw <= eps:
            if self.prioritized: calOut()
            return self.getRndAction()
        else:
            calOut()
            return self.dqnOnline.processAction(output.argmax(1))

    def getRndAction(self):
        return nprnd.randint(self.nActions)

    def update(self, infoDicts):
        st1, st2, rewards, actions, dones, IS = unpackBatch(infoDicts, device = self.device)
        qValues = self.dqnOnline(st1).gather(1, actions)
        qTargets = dqnTarget(self.dqnOnline, self.dqnTarget, 
                                st2, rewards, dones, self.gamma, self.double)

        loss = F.smooth_l1_loss(qValues, qTargets, reduction="none")
        if self.prioritized:
            IS = importanceNorm(IS)
            loss = Tmul(IS, loss)
        loss = Tmean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tbw != None:
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            self.tbw.add_scalar('train/Mean TD Error', torch.mean(qTargets - qValues).item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.eva_meang, self.eva_maxg)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1

    @property
    def device(self):
        return self.dqnOnline.device
