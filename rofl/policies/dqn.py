from rofl.functions.const import *
from rofl.functions.torch import *
from rofl.functions import EpsilonGreedy
from .base import Policy

def dqnTarget(onlineNet, targetNet, s2, r, t, gamma, double:bool = True):
    with no_grad():
        model_out = targetNet.forward(s2)
        if double:
            On_model_out = onlineNet.forward(s2)
            a_greedy = On_model_out.max(1)[1]
            Qs2_max = model_out.gather(1, a_greedy.unsqueeze(1))
        else:
            Qs2_max = model_out.max(1)[0].unsqueeze(1)
        t = t.bitwise_not()
        target = r + Tmul(t, Qs2_max).mul(gamma).reshape(r.shape)
    return target

def importanceNorm(IS):
    IS = IS.unsqueeze(1)
    maxIS = IS.max()
    return IS / maxIS

class dqnPolicy(Policy):
    discrete = True
    name = "dqnPolicy"
    def initPolicy(self, **kwargs):
        config = self.config
        self.dqnOnline = self.actor
        self.dqnTarget = cloneNet(self.actor)

        self.epsilon = EpsilonGreedy(config)
        self.prioritized = config["agent"]["memory_prioritized"]

        self.updateTarget = config["policy"]["freq_update_target"]
        #self.nActions = config["policy"]["n_actions"]
        self.double = config['policy'].get("double", False)
        
        parameters, lr = self.dqnOnline.parameters(), config['policy']["learning_rate"]
        if config['policy']["optimizer"] == "adam":
            self.optimizer = optim.Adam(parameters, lr = lr, **config.get("optimizer_args", {}))
        elif config['policy']["optimizer"] == "rmsprop":
            self.optimizer = optim.RMSprop(parameters, lr = lr, **config.get("optimizer_args", {}))
        
        self.epochs = 0

    def getAction(self, state):
        throw = nprnd.uniform()
        eps = self.epsilon.test(state) if self.test else self.epsilon.train(state)

        self.lastNetOutput = self.actor.getQValues(state) if self.prioritized else None

        if throw <= eps:
            return self.getRndAction()
        else:
            output = self.lastNetOutput if self.prioritized else self.actor.getQValues(state)
            return self.actor.processAction(output.argmax(1))

    def update(self, infoDict):
        st1, st2, rewards = infoDict['observation'], infoDict['next_observation'], infoDict['reward']
        actions, dones = infoDict['action'], infoDict['done']
        IS = None # TODO: add this part for sampling importance
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

        if (self.tbw != None) and (self.epochs % self.tbwFreq == 0):
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            self.tbw.add_scalar('train/Mean TD Error', torch.mean(qTargets - qValues).item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.evalMeanGrad, self.evalMaxGrad)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1

    @property
    def device(self):
        return self.dqnOnline.device
