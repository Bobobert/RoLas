from .base import Agent
from rofl.functions.const import *
from rofl.utils.dqn import MemoryReplay, MemoryReplayFF
from rofl.utils.cv import imgResize, YChannelResize
from rofl.functions.gym import noOpSample
from rofl.functions import runningStat
from rofl.functions.torch import updateNet
from tqdm import tqdm

class dqnAtariAgent(Agent):

    name = "dqnAgentv0"

    def initAgent(self, useTQDM = False, **kwargs):
        config = self.config
        self.done, self.lives = True, None

        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.memory = MemoryReplay(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        LHist= lhist)
        self.obsShape = (lhist, *obsShape)
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)
        self.noOpAction = self._noop_ if self.noOpSteps > 0 else None
        self.frameSize = tuple(obsShape)
        self.isAtari = config["env"]["atari"]
        self.memPrioritized = config["agent"].get("memory_prioritized", False)
        self.tqdm = useTQDM

        self.fixedTrajectory = None
        self.frameStack, self.lastObs, self.lastFrame = np.zeros(self.obsShape, dtype = np.uint8), None, None
        

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        if self.isAtari:
            self.lastFrame = YChannelResize(obs, size = self.frameSize)
        else:
            self.lastFrame = imgResize(obs, size = self.frameSize)
        self.frameStack[0] = self.lastFrame
        newObs = torch.from_numpy(self.frameStack).to(self.device)
        return newObs.unsqueeze(0).float().div(255)

    def getBatch(self, size:int, proportion: float = 1.0):
        """
            Prepares and returns the dictionary with a full or
            a mini batch of the trajectories for the algorithm 
            to update the policy

            returns
            -------
            batch: dict
                At least size elements
        """
        # Generate iterator
        I = range(ceil(size * proportion))
        if self.tqdm:
            I = tqdm(I, desc="Generating batch")
        # Do steps for batch
        self.policy.test = False
        for i in I:
            self.step()
        return self.memory.sample(size, device = self.device, prioritized = self.memPrioritized)

    def step(self, randomPi = False):
        """
        Take a step in the environment
        """
        env, proc, pi = self.env, self.processObs, self.policy
        # Prepare state
        if self.done:
            obs = env.reset()
            # No op, no actions when starting
            if self.noOpSteps > 0:
                for _ in range(random.randint(1, self.noOpSteps)):
                    obs, _, _, info = env.step(self.noOpAction)
                self.lives = info.get("ale.lives", 0)
            obs = proc(obs, True)
        else:
            obs = self.lastObs
        # Take action
        if randomPi:
            action = pi.getRandom()
        else:
            action = pi.getAction(obs)
        nextObs, reward, done, info = env.step(action)
        nextLives = info.get("ale.lives", 0)
        self.reward = reward
        # Clip reward if needed
        if self.clipReward > 0.0:
            reward = np.clip(reward, -self.clipReward, self.clipReward)
        # update memory replay
        markTerminal = done if self.lives == nextLives else True
        self.memory.add(self.lastFrame, action, reward, markTerminal)
        if self.memPrioritized:
            self.memory.addTD(None if randomPi else pi.lastNetOutput)
        # if termination prepare env
        self.done, self.lives = done, nextLives
        if not done:
            self.lastObs = proc(nextObs)

    def reportCustomMetric(self):
        return reportQmean(self)

    def currentState(self):
        """
            Returns a dict with all the required information
            of its state to start over or just to save it.
        """
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the agent.
            Must verify the name of the agent is the same and the
            type.
        """
        return NotImplementedError

def prepare4Ratio(obj):
    obj.ratioTree = runningStat()

def calRatio(obj, env):
    # Calculate ratio from environment
    cc = env.cell_counts
    tot = env.n_col * env.n_row
    obj.ratioTree += cc[env.tree] / tot

def reportQmean(obj):
    if obj.fixedTrajectory is None:
        return 0.0
    with no_grad():
        model_out = obj.policy.dqnOnline(obj.fixedTrajectory)
        mean = Tmean(model_out.max(1).values).item()
    if obj.tbw != None:
        obj.tbw.add_scalar("test/mean max Q", mean, obj.testCalls)
    return mean

def reportRatio(obj):
    meanQ = reportQmean(obj)
    if obj.tbw != None:
        obj.tbw.add_scalar("test/mean tree ratio", obj.ratioTree.mean, obj.testCalls)
        obj.tbw.add_scalar("test/std tree ratio", obj.ratioTree.std, obj.testCalls)
    return {"mean_q": meanQ, 
            "mean tree ratio": obj.ratioTree.mean, 
            "std tree ratio":obj.ratioTree.std}

class dqnFFAgent(dqnAtariAgent):
    name = "dqnForestFireAgentv0"

    def initAgent(self, useTQDM = False, **kwargs):
        # From the original agent
        super(dqnFFAgent, self).initAgent(**kwargs)

        config = self.config
        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        nCol, nRow = config["env"]["n_col"], config["env"]["n_row"]
        if not config["agent"].get("scale_pos", False):
            nCol, nRow = 1, 1
        self.memory = MemoryReplayFF(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        LHist= lhist,
                        nCol = nCol, nRow = nRow)
        self.lastPos = None

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        frame, pos, tm = obs["frame"], obs["position"], obs.get("time", 0)
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        self.lastFrame = imgResize(frame, size = self.frameSize)
        self.frameStack[0] = self.lastFrame
        self.lastFrame = {"frame":self.lastFrame, "position":pos, "time":tm}
        newObs = torch.from_numpy(self.frameStack).to(self.device).unsqueeze(0).float().div(255)
        Tpos = torch.as_tensor(pos).to(self.device).float().unsqueeze(0)
        Ttm = torch.as_tensor([tm]).to(self.device).float().unsqueeze(0)
        return {"frame": newObs, "position":Tpos, "time":Ttm}

    def prepareTest(self):
        prepare4Ratio(self)

    def calculateCustomMetric(self, env, reward, done):
        calRatio(self, env)

    def reportCustomMetric(self):
        return reportRatio(self)

class dqnFFAgent2(dqnAtariAgent):
    name = "dqnForestFireAgentv1"

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        self.lastFrame = obs
        self.frameStack[0] = self.lastFrame
        return torch.from_numpy(self.frameStack).to(self.device).unsqueeze(0).float().div(255)

    def prepareTest(self):
        prepare4Ratio(self)

    def calculateCustomMetric(self, env, reward, done):
        calRatio(self, env)

    def reportCustomMetric(self):
        return reportRatio(self)

class dqnRollout(Agent):
    _EPS_ = 1e-4

    def initAgent(self, heuristic = None, **kwargs):
        self.heuristic = heuristic
        if heuristic is None:
            raise ValueError("Heuristic needs to be not a NoneType")
        config = self.config
        self.ucbC = config["agent"].get("ubc_c", 1.0)
        self.rolloutDepth = config["agent"].get("rollout_depth", 1)
        self.rolloutLength = config["agent"].get("rollout_length", 10)
        self.rolloutSamples = config["agent"].get("rollout_samples", 30)
        self.nActions = config["policy"]["n_actions"]

    def createNT(self):
        """
            Returns a Tensor that has a shape 
            [Episode length, Number actions]
        """
        c = self.config
        l = c["env"].get("max_length", 1024)
        n = c["policy"]["n_actions"]
        return torch.zeros((l, n), dtype = I_TDTYPE_DEFT)

    def startRollout(self, state, action, nt):
        
        rt = 0.0 # mean R_t
        gs = 0.0

        for i in range(self.rolloutSamples):
            # Reset state
            self.loadState(state)
            obs0 = self.envStep(action)
            # mean Rt
            rt = runningMean(rt, obs0["reward"], i)
            rs, gm, keep, d = obs0["reward"], self.gamma, True, 0

            if obs0["done"]: 
                keep = False

            while d < self.rolloutLength and keep:
                # Do the UCB actions
                if d < self.rolloutDepth:
                    # take UCB action
                    At = self.UCBAction(self._envStep_, nt[self._envStep_])
                    nt[self._envStep_, At] += 1
                else:
                    # Use heuristic
                    At = self.heuristic(self.lastObs)
                obsDict = self.envStep(At)
                rs += gm * obsDict["reward"]
                d += 1
                gm *= self.gamma
                # Stop condition for the actual rollout
                if obsDict["done"]:
                    keep = False
            # Bootstrap last value with dqn
            if keep:
                qvalues = self.getQvalues(self.lastObs)
                rs += gm * qvalues.max().item()
            gs = runningMean(gs, rs, i)
            # TODO insert time / resources limit stop condition here

        return gs, rt, nt
            
    def UCBAction(self, t, nt):
        """
            Consider nt as the array shape [n_actions,], that holds
            how many times the action have been seen at time t.

            parameters
            ----------
            t: int
                step relative to the Tensor nt
            nt: Tensor
                Should have how manytimes an action has been seen
                at the time t
        """
        qvalues = self.getQvalues(self.lastObs)
        lnt = qvalues.new_empty(qvalues.shape).fill_(math.log(t))
        qs = torch.addcdiv(qvalues, self.ucbC, lnt, nt + self._EPS_)

        return qs.argmax().item()
        
    def getQvalues(self, obs):
        return self.policy.dqnOnline.forward(obs).squeeze()

    def updateDQN(self, netParameters):
        updateNet(self.policy.dqnOnline, netParameters)
    