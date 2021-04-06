from rofl.functions.const import *
from rofl.functions.torch import *
from rofl.functions import EpsilonGreedy
from .base import Policy

def unpackBatch(*dicts, device = DEVICE_DEFT):
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones  = [], [], [], [], []
        for trajectory in dicts:
            states1 += [trajectory["st"]]
            states2 += [trajectory["st1"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
        st1 = Tcat(states1, dim=0).to(device)
        st2 = Tcat(states2, dim=0).to(device)
        actions = Tcat(actions, dim=0).to(device)
        rewards = Tcat(rewards, dim=0).to(device)
        dones = Tcat(dones, dim=0).to(device)
    else:
        trajectoryBatch = dicts[0]
        st1 = trajectoryBatch["st"].to(device)
        actions = trajectoryBatch["action"].to(device).long()
        rewards = trajectoryBatch["reward"].to(device)
        st2 = trajectoryBatch["st1"].to(device)
        dones = trajectoryBatch["done"].to(device)
    actions = actions.unsqueeze(1)
    return st1, st2, rewards, actions, dones

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

class dqnPolicy(Policy):
    discrete = True
    name = "dqnPolicy"
    def __init__(self, config, dqn, tbw = None):

        self.config = config.copy()
        self.dqnOnline = dqn
        self.dqnTarget = cloneNet(dqn)

        self.gamma = config["agent"]["gamma"]
        config = config["policy"]
        self.updateTarget = config["freq_update_target"]
        self.nActions = config["n_actions"] - 1
        self.double = config.get("double", False)
        self.epsilon = EpsilonGreedy(config["epsilon_start"],
                                    config["epsilon_end"],
                                    config["epsilon_life"],
                                    "linear",
                                    config["epsilon_test"])
        
        parameters, lr = self.dqnOnline.parameters(), config["learning_rate"]
        if config["optimizer"] == "adam":
            self.optimizer = optim.Adam(parameters, lr = lr)
        elif config["optimizer"] == "rmsprop":
            self.optimizer = optim.RMSprop(parameters, lr = lr)
        
        self.epochs = 0
        self.tbw = tbw
        super(dqnPolicy, self).__init__()

    def getAction(self, state):
        throw = random.uniform(0,1)
        eps = self.epsilon.test(state) if self.test else self.epsilon.train(state)
        if throw <= eps:
            return self.getRandom()
        else:
            return self.dqnOnline.getAction(state)

    def getRandom(self):
        return random.randint(0, self.nActions)

    def update(self, *infoDicts):
        st1, st2, rewards, actions, dones = unpackBatch(*infoDicts, device = self.device)
        qValues = self.dqnOnline(st1)
        qValues = qValues.gather(1, actions) # maybe this blows
        qTargets = dqnTarget(self.dqnOnline, self.dqnTarget, 
            st2, rewards, dones, self.gamma)

        loss = F.smooth_l1_loss(qValues, qTargets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tbw is not None:
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            #self.tbw.add_scalar('train/Mean TD Error', torch.mean(q_target - q_online).item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1

    @property
    def device(self):
        return self.dqnOnline.device

    def currentState(self):
        """
            Returns a dict with all the required information
            of its state to start over or just to save it.
        """
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the policy.
            Must verify the name of the policy is the same and the
            type.
        """
        return NotImplementedError