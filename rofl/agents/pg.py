from .base import Agent
from rofl.functions.const import DEVICE_DEFT
from rofl.functions.functions import np, torch, rnd, no_grad, ceil
from rofl.functions.torch import array2Tensor
from rofl.utils.pg import Memory, MemoryFF
from rofl.functions.coach import singlePathRollout
from rofl.utils.openCV import imgResize

class pgAgentNOT(Agent):
    name = "pg_agent_v0" #TO BE DELETED

    def initAgent(self, **kwargs):

        config = self.config
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)
    
        self.memory = Memory(config)

    def processObs(self, obs, reset=False):
        return array2Tensor(obs, device = self.policy.device)

    def singlePath(self):
        # Init from last
        env = self.env
        obs = self.obs
        delta = 0.0
        endBySteps = False
        stepsDone = 0
        pi, baseline = self.policy.actor, self.policy.baseline
        proc = self.processObs

        # Reseting env and variables
        if self.done:
            obs = env.reset()
            # No op, no actions when starting
            if self.noOpSteps > 0:
                for _ in range(rnd.randint(1, self.noOpSteps)):
                    obs, _, _, info = env.step(self.noOp)
            obs = proc(obs, True)
            self.done = False
        if baseline is not None:
            baselineObs = baseline.getValue(obs)
        else:
            baselineObs = 0.0
        
        # Env Loop
        while True:
            stepsDone += 1
            with no_grad():
                action, log_action, _ = pi.sampleAction(pi(obs))
                nextObs, reward, done, _ = env.step(pi.processAction(action))
                nextObs = proc(nextObs)
                reward = float(reward)
                # Clip reward if needed
                if self.clipReward > 0.0:
                    reward = np.clip(reward, -self.clipReward, self.clipReward)
                # Baseline calculation
                if baseline is not None:
                    nextBaseline = baseline.getValue(nextObs) if not done else 0.0
                else:
                    nextBaseline = False
            if self.gae:
                # Calculate delta_t
                delta = reward + self.gamma * nextBaseline - baselineObs
            # Enough steps, can do bootstrapping for the last value
            if (stepsDone == self.maxEpLen) and (baseline is not None):
                reward += self.gamma * nextBaseline
                endBySteps = True
            self.memory.add(obs, action, log_action, reward, done or endBySteps, advantage=delta)
            obs = nextObs
            baselineObs = nextBaseline
            # End of loop
            if done:
                self.done = True
                self.episodes += 1
                break
            elif endBySteps:
                break
        
        self.obs = obs
    
    def getBatch(self, size:int, proportion:float = 1.0):
        minSample = ceil(size * proportion) if size > 0 else 1
        self.policy.test = False
        while len(self.memory) < minSample:
            self.singlePath()
        sample = self.memory.sample(minSample if size > 0 else -1, device = self.device)
        self.memory.clean()
        return sample

class pgAgent(Agent):
    def initAgent(self, **kwargs):
        config = self.config
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)

    def processReward(self, reward):
        return np.clip(reward, -self.clipReward, self.clipReward)

    def processObs(self, obs):
        return array2Tensor(obs, device = self.policy.device)

    def getEpisode(self, random):
        return singlePathRollout(self, self.maxEpLen, random)


class pgFFAgent(pgAgent):
    name = "forestFire_pgAgent"
    def __init__(self, config, policy, envMaker, tbw = None):

        super(pgFFAgent, self).__init__(config, policy, envMaker, tbw)
        self.isAtari = config["env"]["atari"]
        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.memory = MemoryFF(config)
        self.obsShape = (lhist, *obsShape)
        self.frameSize = obsShape
        self.frameStack, self.lastObs, self.lastFrame = np.zeros(self.obsShape, dtype = np.uint8), None, None
        
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