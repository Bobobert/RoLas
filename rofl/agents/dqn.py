from .base import Agent
from rofl.functions.const import *
from rofl.utils.dqn import *
from rofl.utils.cv import imgResize, YChannelResize

class dqnAtariAgent(Agent):

    name = "dqnAgentv0"

    def initAgent(self, useTQDM = False, **kwargs):
        config = self.config
        self.done, self.lives = True, None

        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.memory = MemoryReplay(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        LHist= lhist)
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)
        self.noOpAction = self.noOp if self.noOpSteps > 0 else None
        self.frameSize = tuple(obsShape)
        self.isAtari = config["env"]["atari"]
        self.memPrioritized = config["agent"].get("memory_prioritized", False)
        self.tqdm = useTQDM

        self.fixedTrajectory = None
        self.frameStack, self.lastObs, self.lastFrame = genFrameStack(config), None, None
        

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        if self.isAtari:
            obs = self.lastFrame = YChannelResize(obs, size = self.frameSize)
        else:
            obs = self.lastFrame = imgResize(obs, size = self.frameSize)
        return lHistObsProcess(self, obs, reset)

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
            from tqdm import tqdm
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
                for _ in range(rnd.randint(1, self.noOpSteps)):
                    obs, _, _, info = env.step(self.noOpAction)
                self.lives = info.get("ale.lives", 0)
            obs = proc(obs, True)
        else:
            obs = self.lastObs
        # Take action
        if randomPi:
            action = pi.getRndAction()
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
