from .base import Agent
from rofl.functions.const import *
from rofl.functions.gym import noOpSample
from tqdm import tqdm

import cv2
def imgResize(f, size = FRAME_SIZE):
    return cv2.resize(f, size)

def YChannelResize(f, size = FRAME_SIZE):
    f = cv2.cvtColor(f, cv2.COLOR_RGB2YUV)[:,:,0]
    return imgResize(f, size)

class MemoryReplay(object):
    """
    Main Storage for the transitions experienced by the actors.

    It has methods to Sample

    Parameters
    ----------
    capacity: int
        Number of transitions to store
    """
    def __init__(self,
                 capacity:int = MEMORY_SIZE,
                 state_shape:list = FRAME_SIZE,
                 LHist:int = LHIST,
                 state_dtype_in:np.dtype = np.uint8,
                 state_dtype_out:np.dtype = np.float32,
                 action_dtype_in:np.dtype = np.uint8,
                 action_dtype_out:torch.dtype = torch.int64,
                 reward_dtype_in:np.dtype = np.float32,
                 reward_dtype_out:torch.dtype = torch.float32,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in
        self.s_dtype = state_dtype_out
        self.a_dtype = action_dtype_out
        self.r_dtype = reward_dtype_out

        self.capacity = capacity
        self.LHist = LHist
        self.shapeHistOut = [LHist] + list(state_shape)
        self._i = 0
        self.FO = False

        self.s_buffer = np.zeros([capacity] + list(state_shape), dtype = state_dtype_in)
        self.a_buffer = np.zeros(capacity, dtype = action_dtype_in)
        self.r_buffer = np.zeros(capacity, dtype = reward_dtype_in)
        self.t_buffer = np.ones(capacity, dtype = np.bool_) # Inverse logic

    def add(self, s, a, r, t):
        """
        Add one item
        """
        self.s_buffer[self._i] = s
        self.a_buffer[self._i] = a
        self.r_buffer[self._i] = r
        self.t_buffer[self._i] = t
        self._i = (self._i + 1) % self.capacity
        if self._i == 0:
            self.FO = True

    def get2History(self, i:int, m:int, st1, st2):
        # modify inplace
        for n, j in enumerate(range(i, i - self.LHist - 1, -1)):
            s, _, _, t = self[j]
            if n < self.LHist:
                st2[m][n] = s
            if n > 0:
                st1[m][n - 1] = s
            if not t and n >= 0:
                # This should happend rarely
                break

    def __getitem__(self, i:int):
        if i < self._i or self.FO:
            i = i % self.capacity
            return (self.s_buffer[i],
                    self.a_buffer[i],
                    self.r_buffer[i],
                    self.t_buffer[i])
        else:
            return self.zeroe

    @property
    def zeroe(self):
        return (np.zeros(self.s_in_shape, dtype=self.s_dtype_in),
                0,
                0.0,
                False)

    def sample(self, mini_batch_size:int):
        """
        Process and returns a mini batch. The tuple returned are
        all torch tensors.
        
        If device is cpu class, this process may consume more cpu resources
        than expected. Could be detrimental if hosting multiple instances. 
        This seems expected from using torch. (Y)

        Parameters
        ---------
        mini_batch_size: int
            Number of samples that compose the mini batch
        device: torch.device
            Optional. Torch device target for the mini batch
            to reside on.
        """
        assert mini_batch_size > 0, "The size of the mini batch must be positive"

        if self._i > mini_batch_size + self.LHist or self.FO:
            ids = np.random.randint(self.LHist, self.capacity if self.FO else self._i - 1, 
                                    size=mini_batch_size)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = self.s_dtype)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i, i - self.LHist - 1, -1)):
                    s, _, _, t = self[j]
                    if n < self.LHist:
                        st2[m][n] = s.copy()
                    if n > 0:
                        st1[m][n - 1] = s.copy()
                    if not t and n >= 0:
                        # This should happend rarely
                        break
            at = self.a_buffer[ids]
            rt = self.r_buffer[ids]
            terminals = self.t_buffer[ids].astype(np.float32)
            # Passing to torch format
            st1 = torch.as_tensor(st1).div(255).requires_grad_()
            st2 = torch.as_tensor(st2).div(255)
            terminals = torch.as_tensor(terminals, dtype=torch.float32)
            at = torch.as_tensor(at, dtype=self.a_dtype)
            rt = torch.as_tensor(rt, dtype=self.r_dtype)
            return {"st":st1,"st1":st2, "reward": rt, "action":at, "done":terminals}
        else:
            raise IndexError("The memory does not contains enough transitions to generate the sample")

    def __len__(self):
        if self.FO:
            return self.capacity
        else:
            return self._i

    def showBuffer(self, samples:int = 20, Wait:int = 3):
        import matplotlib.pyplot as plt
        # Drawing samples
        Samplei = np.random.randint(self._i if not self.FO else self.capacity, size=samples)
        for i in Samplei:
            plt.ion()
            fig = plt.figure(figsize=(10,3))
            plt.title('Non-terminal' if self.t_buffer[i] else 'Terminal')
            plt.axis('off')
            for n, j in enumerate(range(i, i - self.LHist, -1)):
                fig.add_subplot(1, self.LHist, n + 1)
                plt.imshow(self.s_buffer[j])
                plt.axis('off')
            plt.pause(Wait)
            plt.close(fig)

class dqnAtariAgent(Agent):

    name = "dqnAgentv0"

    def __init__(self, config, policy, envMaker,
                    seedEnv = 1, seedEnvTest:int = 10,
                    tbw = None, useTQDM = False):

        self.config = config.copy()
        self.policy = policy
        self.device = policy.device
        self.env, _ = envMaker(seedEnv)
        self.envTest, _ = envMaker(seedEnvTest)
        self.done = True
        try:
            self.environment = self.env.name
        except:
            self.environment = "unknown"

        self.gamma = config["agent"]["gamma"]
        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.memory = MemoryReplay(capacity=config["agent"]["memory_size"],
                        state_shape = obsShape,
                        LHist= lhist)
        self.obsShape = (lhist, obsShape[0], obsShape[1])
        self.clipReward = config["agent"].get("clip_reward", 0.0)
        self.noOpSteps = config["agent"].get("no_op_start", 0)
        self.noOpAction = noOpSample(self.envTest) if self.noOpSteps > 0 else None
        self.frameSize = obsShape
        self.tbw, self.tqdm = tbw, useTQDM

        self.fixedTrajectory = None
        self.frameStack, self.lastObs, self.lastFrame = np.zeros(self.obsShape, dtype = F_NDTYPE_DEFT), None, None
        super(dqnAtariAgent, self).__init__()

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        if reset:
            self.frameStack.fill(0.0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)
        self.lastFrame = YChannelResize(obs, size = self.frameSize)
        self.frameStack[0] = self.lastFrame
        newObs = torch.from_numpy(self.frameStack).to(self.device)
        return newObs.unsqueeze(0).div(255)

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
        return self.memory.sample(size)

    def step(self, randomPi = False):
        """
        Take a step in the environment
        """
        env, proc, pi = self.env, self.processObs, self.policy
        # Prepare state
        if self.done:
            frame = env.reset()
            # No op, no actions when starting
            if self.noOpSteps > 0:
                for _ in range(random.randint(1, self.noOpSteps)):
                    frame, _, _, _ = env.step(self.noOpAction)
            obs = proc(frame, True)
        else:
            obs = self.lastObs
        frame = self.lastFrame
        # Take action
        if randomPi:
            action = pi.getRandom()
        else:
            action = pi.getAction(obs)
        nextFrame, reward, done, _ = env.step(action)
        # Clip reward if needed
        if self.clipReward > 0.0:
            reward = np.clip(reward, -self.clipReward, self.clipReward)
        # update memory replay
        self.memory.add(frame, action, reward, not done)
        # if termination prepare env
        self.done = done
        if not done:
            self.lastObs = proc(nextFrame)

    def reportCustomMetric(self):
        if self.fixedTrajectory is None:
            return 0.0
        with no_grad():
            model_out = self.policy.dqnOnline(self.fixedTrajectory)
            mean = torch.mean(model_out.max(1).values).item()
        if self.tbw is not None:
            self.tbw.add_scalar("test/mean max Q", mean, self.testCalls)
        return mean

    def reset(self):
        self.done = True
        self.memory._i, self.memory.FO = 0, False

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
