from rofl.policies import DqnPolicy, BasePolicy
from rofl.functions.torch import *
from rofl.functions.const import *

class simpleDQN(BasePolicy):
    """
        Simple DQN container
    """
    name = "simple dqn policy"
    def __init__(self, config, dqn):
        self.config = config
        self.dqnOnline = dqn

    def new(self):
        return simpleDQN(self.config, cloneNet(self.dqnOnline))

    def currentState(self):
        return None

    def loadState(self, newState):
        pass

    @property
    def device(self):
        return self.dqnOnline.device

class DqnRollPolicy(DqnPolicy):
    name = "dqn rollout policy"

    def __init__(self, config, net, envMaker, agentClass, tbw = None):
        super().__init__(config, net, tbw)
        nagents = config["policy"].get("n_agents", NCPUS)
        if nagents > self.nActions:
            nagents = self.nActions
        config["agent"]["workers"] = nagents

        from rofl.agents.dqnrollout import multiAgentRollout

        pi = simpleDQN(config, net)
        self.master = multiAgentRollout(config, pi, envMaker, agentClass)
        self.nt = None
        self.dqnTarget = None #TODO ?

    def new(self):
        # Comodin
        return simpleDQN(self.config, cloneNet(self.dqnOnline))

    def currentState(self):
        pass

    def loadState(self, newState):
        pass

    def getRollAction(self, agentState):
        if self.nt is None:
            raise ValueError("Policy need a N_t tensor in nt in order to work")
        action, self.nt, qs, rt = self.master.rollout(agentState, self.nt)

        return action, qs, rt

    def update(self, infoDict):
        st = infoDict["observation"].detach_().requires_grad_()
        qTargets = infoDict["q_values"]

        qValues = self.dqnOnline(st)

        loss = F.smooth_l1_loss(qValues, qTargets, reduction="mean")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tbw != None:
            self.tbw.add_scalar('train/Loss', loss.item(), self.epochs)
            self.tbw.add_scalar('train/Mean TD Error', torch.mean(qTargets - qValues).item(), self.epochs)
            max_g, mean_g = analysisGrad(self.dqnOnline, self.evalMeanGrad, self.evalMaxGrad)
            self.tbw.add_scalar("train/max grad",  max_g, self.epochs)
            self.tbw.add_scalar("train/mean grad",  mean_g, self.epochs)
        if self.epochs % self.updateTarget == 0:
            # Updates the net
            updateNet(self.dqnTarget, self.dqnOnline.state_dict())
        self.epochs += 1

    def syncParameters(self):
        params = getListState(self.dqnOnline)
        free, working, _ = self.master.listWorkers()
        if working != []:
            print("Warning: Actors from multiActor have working status")
        for w in free:
            w.ref = w().updateDQN.remote(params)
        self.master.syncResolve(free)
