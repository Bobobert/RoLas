from .base import Agent
from rofl.functions.const import *
from rofl.utils.dqn import MemoryReplay, MemoryReplayFF
from rofl.utils.cv import imgResize, YChannelResize
from rofl.functions.gym import noOpSample
from tqdm import tqdm

class dqnAtariAgent(Agent):

    name = "dqnAgentv0"

    def __init__(self, config, policy, envMaker,
                    tbw = None, useTQDM = False):

        self.config = config.copy()
        self.policy = policy
        self.device = policy.device
        self.env, _ = envMaker(config["env"]["seedTrain"])
        self.envTest, _ = envMaker(config["env"]["seedTest"])
        self.done, self.lives = True, None
        try:
            self.environment = self.env.name
        except:
            self.environment = "unknown"

        self.gamma = config["agent"]["gamma"]
        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.memory = MemoryReplay(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        LHist= lhist)
        self.obsShape = (lhist, *obsShape)
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)
        self.noOpAction = noOpSample(self.envTest) if self.noOpSteps > 0 else None
        self.frameSize = tuple(obsShape)
        self.isAtari = config["env"]["atari"]
        self.memPrioritized = config["agent"].get("memory_prioritized", False)
        self.tbw, self.tqdm = tbw, useTQDM

        self.fixedTrajectory = None
        self.frameStack, self.lastObs, self.lastFrame = np.zeros(self.obsShape, dtype = np.uint8), None, None
        super(dqnAtariAgent, self).__init__()

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

    def prepareTest(self):
        """
            If the agent needs to prepare to save or load 
            anything before a test. Write it here
        """
        #self.frameStack = None
        None
    
    def prepareAfterTest(self):
        """
            If the agent needs to prepare to save or load 
            anything after a test. Write it here
        """
        #self.frameStack = None
        self.done = True

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
        # Clip reward if needed
        if self.clipReward > 0.0:
            reward = np.clip(reward, -self.clipReward, self.clipReward)
        # update memory replay
        markTerminal = done if self.lives == nextLives else True
        self.memory.add(self.lastFrame, action, reward, markTerminal)
        self.memory.addTD(None if randomPi else pi.lastNetOutput)
        # if termination prepare env
        self.done, self.lives = done, nextLives
        if not done:
            self.lastObs = proc(nextObs)

    def reportCustomMetric(self):
        if self.fixedTrajectory is None:
            return 0.0
        with no_grad():
            model_out = self.policy.dqnOnline(self.fixedTrajectory)
            mean = torch.mean(model_out.max(1).values).item()
        if self.tbw != None:
            self.tbw.add_scalar("test/mean max Q", mean, self.testCalls)
        return mean

    def reset(self):
        self.done = True
        self.memory.reset()

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

class dqnFFAgent(dqnAtariAgent):
    name = "dqnForestFireAgentv0"
    def __init__(self, config, policy, envMaker,
                    tbw = None, useTQDM = False):
        # From the original agent
        super(dqnFFAgent, self).__init__(config, policy, envMaker,
                                            tbw = tbw, useTQDM=useTQDM)

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
        frame, pos = obs["frame"], obs["position"]
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        self.lastFrame = imgResize(frame, size = self.frameSize)
        self.frameStack[0] = self.lastFrame
        self.lastFrame = {"frame":self.lastFrame, "position":pos}
        newObs = torch.from_numpy(self.frameStack).to(self.device).unsqueeze(0).float().div(255)
        Tpos = torch.as_tensor(pos).to(self.device).float().unsqueeze(0)
        return {"frame": newObs, "position":Tpos}

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
