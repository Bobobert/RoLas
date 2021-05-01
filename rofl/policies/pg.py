from .base import Policy
from rofl.functions.const import *
from rofl.functions.gym import assertActionSpace
from rofl.functions.torch import *
from rofl.utils.pg import BASELINE_CONFIG_DEFT, unpackBatch

class pgPolicy(Policy):
    """
        Vanilla REINFORCE policy gradient policy.
        Can handle and actor and a baseline.
        Update method expects results from an episodic
        task, therefore gae is not used.
    """
    name = "pgPolicyv0"
    def __init__(self, config, actor, baseline = None, tbw = None):
        self.actor = actor
        self.baseline = baseline
        
        self.AS = assertActionSpace(config)
        
        # Config actor
        pconfig = config["policy"]
        parameters, lr = actor.parameters(), pconfig["learning_rate"]
        if pconfig["optimizer"] == "adam":
            self.optimizer = optim.Adam(parameters, lr = lr)
        elif pconfig["optimizer"] == "rmsprop":
            self.optimizer = optim.RMSprop(parameters, lr = lr)
        
        self.beta = config["policy"].get("entropy_bonus", 0.0)
        # Config Baseline
        bconfig = config.get("baseline", BASELINE_CONFIG_DEFT)
        if baseline != None:
            parameters, lr = baseline.parameters(), bconfig["learning_rate"]
            if bconfig["optimizer"] == "adam":
                self.blOpt = optim.Adam(parameters, lr = lr)
            elif bconfig["optimizer"] == "rmsprop":
                self.blOpt = optim.RMSprop(parameters, lr = lr)

        # Administrative
        self.config, self.epochs = config, 0
        self.tbw = tbw
        self.eva_maxg = pconfig["evaluate_max_grad"]
        self.eva_meang = pconfig["evaluate_mean_grad"]
        self.clipGrad = pconfig.get("clip_grad", 0.0)

    def getAction(self, state):
        return self.actor.getAction(state)

    def getRandom(self):
        return self.AS.sample()

    @property
    def device(self):
        return self.actor.device

    def update(self, *infoDicts):
        states, actions, returns, logprobs = unpackBatch(*infoDicts, device = self.device)
        # Setting Actor update
        out = self.actor.forward(states)
        dist = self.actor.getDist(out)
        logActions, lossEntropy = dist.log_prob(actions.detach_()), 0.0
        if self.beta > 0.0:
            entropy = dist.entropy()
            lossEntropy = - self.beta * Tmean(entropy)
        # Calculate advantage with baseline
        if self.baseline != None:
            bStates = cloneState(states, False)
            baselines = self.baseline.forward(bStates)
            advantages = returns - baselines.squeeze()
        else:
            baselines = None
            advantages = returns
        # Update actor
        self.optimizer.zero_grad()
        if len(logActions.shape) > 1:
            if logActions.shape[1] > 1:
                logActions = Tsum(logActions, dim = -1)
        loss = -1.0 * Tmean(Tmul(logActions, advantages)) + lossEntropy
        loss.backward()
        if self.clipGrad > 0.0:
            clipGrads(self.actor, self.clipGrad)
        self.optimizer.step()

        self.updateBaseline(states, returns)

        if self.tbw != None:
            self.tbw.add_scalar('train/Actor loss', loss.item(), self.epochs)
            max_g, mean_g = analysisGrad(self.actor, self.eva_meang, self.eva_maxg)
            self.tbw.add_scalar("train/Actor max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/Actor mean grad",  mean_g, self.epochs)
            #if self.baseline != None:
            #    self.tbw.add_scalar("train/Baseline loss", lossBl.item(), self.epochs)
        
        self.epochs += 1

    def updateBaseline(self, states, returns):
        if self.baseline is None:
            return None

        minibatchSize = self.config["baseline"]["minibatch_size"]
        minibatches = self.config["baseline"]["batch_minibatches"]

        def backLoss(a, b, retain = False):
            self.blOpt.zero_grad()
            loss = F.mse_loss(a.squeeze(), b)
            loss.backward(retain_graph = retain)
            self.blOpt.step()
            return loss

        if minibatchSize >= returns.shape[0]:
            baselines = self.baseline(cloneState(states, True))
            loss = backLoss(baselines, returns)
            loss = loss.item()
        else:
            loss = 0
            for _ in range(minibatches):
                iDx = np.random.randint(0, returns.shape[0], size=minibatchSize)
                baselines = self.baseline(cloneState(states, True, iDx))
                loss += backLoss(baselines, returns[iDx])
            loss = loss / minibatches

        if self.tbw != None:
            self.tbw.add_scalar("train/Baseline loss", loss, self.epochs)