from rofl.functions.const import *
from rofl.functions import runningStat

def genFrameStack(config):
    return np.zeros((config['agent']['lhist'], *config['env']['obs_shape']), dtype = np.uint8)

def lHistObsProcess(agent, obs, reset):
    """
        From an already processed observation image type, modifies the
        lhist framestack.
        Agent is supossed to have a frameStack atribute.
    """
    try:
        framestack = agent.frameStack
    except AttributeError:
        agent.frameStack = framestack = genFrameStack(agent.config)
        print("Warning: obs invalid, agent didn't had frameStack declared") # TODO: add debuger level
    if reset:
        framestack.fill(0)
    else:
        framestack = np.roll(framestack, 1, axis = 0)
    framestack[0] = obs
    agent.frameStack = framestack

    return torch.from_numpy(framestack).unsqueeze(0).to(agent.device).float().div(255)

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
                 action_dtype_in:np.dtype = np.uint8,
                 reward_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in

        self.capacity, self.epsTD = capacity, 1 / capacity
        self.LHist = LHist
        self.shapeHistOut = [LHist] + list(state_shape)
        self._i = 0
        self.FO = False
        self.lastPartialTD, self.sumTD = None, 0.0

        self.s_buffer = np.zeros([capacity] + list(state_shape), dtype = state_dtype_in)
        self.a_buffer = np.zeros(capacity, dtype = action_dtype_in)
        self.r_buffer = np.zeros(capacity, dtype = reward_dtype_in)
        self.t_buffer = np.ones(capacity, dtype = np.bool_) # Inverse logic
        self.e_buffer = np.zeros(capacity, dtype = F_NDTYPE_DEFT)

    def add(self, s, a, r, t):
        """
        Add one item
        """
        i = self._i
        self.s_buffer[i] = s
        self.a_buffer[i] = a
        self.r_buffer[i] = r
        self.t_buffer[i] = not t
        self.e_buffer[i] = 0.0
        self._i = (i + 1) % self.capacity
        if self._i == 0:
            self.FO = True

    def addTD(self, qValues, gamma:float = 1.0):
        # qValues for the i-th state seen
        i = self._i - 1
        t = 1.0 * (not self.t_buffer[i - 1])
        if qValues is None:
            self.e_buffer[i-1] = self.epsTD
        else:
            qValues = qValues.squeeze()
            maxQ = qValues.max().item()
            if self.lastPartialTD is not None:
                TD = abs(self.lastPartialTD + t * gamma * maxQ) + self.epsTD
                self.sumTD += TD - self.e_buffer[i - 1]
                self.e_buffer[i - 1] = TD
            action = self.a_buffer[i]
            actualQ = qValues[action].item()
            self.lastPartialTD = self.r_buffer[i] - actualQ

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

    def getIDS(self, size, prioritized = False):
        s = self.capacity if self.FO else self._i - 1
        if not prioritized:
            ids = nprnd.randint(self.LHist, self.capacity - 1 if self.FO else self._i - 2, 
                                    size=size)
            ps = np.ones(size, dtype = F_NDTYPE_DEFT)
        else:
            a = np.arange(s)
            totTD = self.sumTD if self.FO else self.sumTD - self.e_buffer[s]
            ps = self.e_buffer[a]
            ps = ps / np.sum(ps)
            ids = nprnd.choice(a, size=size, p = ps, replace = True)
            ps = 1 / ps[ids] / s
        return ids, ps

    def sample(self, mini_batch_size:int, device = DEVICE_DEFT, prioritized: bool = False):
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
            ids, ps = self.getIDS(mini_batch_size, prioritized)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = F_NDTYPE_DEFT)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i + 1, i - self.LHist, -1)):
                    s, _, _, t = self[j]
                    if n < self.LHist:
                        st2[m][n] = s
                    if n > 0:
                        st1[m][n - 1] = s
                    if not t and n >= 0:
                        # This should happend rarely
                        break

            # Passing to torch format
            st1 = torch.from_numpy(st1).to(device).div(255).detach_().requires_grad_()
            st2 = torch.from_numpy(st2).to(device).div(255)
            terminals = torch.from_numpy(self.t_buffer[ids]).to(device).float()
            at = torch.from_numpy(self.a_buffer[ids]).to(device).long()
            rt = torch.from_numpy(self.r_buffer[ids]).to(device).float()
            ps = torch.from_numpy(ps).to(device)
            return {"st":st1,"st1":st2, "reward": rt, "action":at, "done":terminals, "IS":ps}
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
        Samplei = nprnd.randint(self._i if not self.FO else self.capacity, size=samples)
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
    
    def reset(self):
        self._i = 0
        self.FO = False
        self.t_buffer[:] = 1
        self.e_buffer[:] = 0.0
        self.lastPartialTD, self.sumTD = None, 0.0

class MemoryReplayFF(MemoryReplay):
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
                 pos_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 action_dtype_in:np.dtype = np.uint8,
                 reward_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 nCol: int = 1, nRow: int = 1,
                 ):
        super().__init__(capacity, state_shape, LHist,
                state_dtype_in, action_dtype_in, reward_dtype_in)
        self.p_buffer = np.zeros([self.capacity] + [2], dtype = pos_dtype_in)
        self.tm_buffer = np.zeros([self.capacity] + [1], dtype = pos_dtype_in)
        self.nCol, self.nRow = nCol, nRow

    def add(self, s, a, r, t):
        """
        Add one item
        """
        s, p, tm = s["frame"], s["position"], s.get("time",0)
        super().add(s, a, r, t)
        self.p_buffer[self._i] = (p[0] / self.nRow, p[1] / self.nCol)
        self.tm_buffer[self._i] = tm

    def __getitem__(self, i:int):
        if i < self._i or self.FO:
            i = i % self.capacity
            return (self.s_buffer[i],
                    self.p_buffer[i],
                    self.tm_buffer[i],
                    self.a_buffer[i],
                    self.r_buffer[i],
                    self.t_buffer[i])
        else:
            return self.zeroe

    @property
    def zeroe(self):
        return (np.zeros(self.s_in_shape, dtype=self.s_dtype_in),
                (0,0),
                0,
                0,
                0.0,
                False)

    def sample(self, mini_batch_size:int, device = DEVICE_DEFT, prioritized: bool = False):
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
            ids, ps = self.getIDS(mini_batch_size, prioritized)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = F_NDTYPE_DEFT)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i + 1, i - self.LHist, -1)):
                    s, _, _, _, _, t = self[j]
                    if n < self.LHist:
                        st2[m][n] = s.copy()
                    if n > 0:
                        st1[m][n - 1] = s.copy()
                    if not t and n >= 0:
                        # This should happend rarely
                        break

            # Passing to torch format
            st1 = torch.from_numpy(st1).to(device).div(255).detach_().requires_grad_()
            st2 = torch.from_numpy(st2).to(device).div(255)
            pos1 = torch.from_numpy(self.p_buffer[ids]).to(device).float()
            pos2 = torch.from_numpy(self.p_buffer[ids + 1]).to(device).float()
            tm1 = torch.from_numpy(self.tm_buffer[ids]).to(device).float()
            tm2 = torch.from_numpy(self.tm_buffer[ids + 1]).to(device).float()
            st1, st2 = {"frame":st1, "position":pos1, "time":tm1}, {"frame":st2, "position":pos2, "time":tm2}
            terminals = torch.from_numpy(self.t_buffer[ids]).to(device).float()
            at = torch.from_numpy(self.a_buffer[ids]).to(device).long()
            rt = torch.from_numpy(self.r_buffer[ids]).to(device).float()
            ps = torch.from_numpy(ps).to(device)
            return {"st":st1,"st1":st2, "reward": rt, "action":at, "done":terminals, "IS": ps, 
                    "obsType":"framePos"}
        else:
            raise IndexError("The memory does not contains enough transitions to generate the sample")

def unpackBatch(*dicts, device = DEVICE_DEFT):
    key = dicts[0].get("obsType", "standard")
    if key == "framePos":
        return unpackBatchComplexObs(*dicts, device = device)
    ### Standard Unpack
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones, IS  = [], [], [], [], [], []
        for trajectory in dicts:
            states1 += [trajectory["st"]]
            states2 += [trajectory["st1"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
            IS += [trajectory["IS"]]
        st1 = Tcat(states1, dim=0)
        st2 = Tcat(states2, dim=0)
        actions = Tcat(actions, dim=0)
        rewards = Tcat(rewards, dim=0)
        dones = Tcat(dones, dim=0)
        IS = Tcat(IS)
    else:
        trajectoryBatch = dicts[0]
        st1 = trajectoryBatch["st"]
        actions = trajectoryBatch["action"]
        rewards = trajectoryBatch["reward"]
        st2 = trajectoryBatch["st1"]
        dones = trajectoryBatch["done"]
        IS = trajectoryBatch["IS"]

    st1 = st1.to(device)
    st2 = st2.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    actions = actions.to(device).unsqueeze(1).long()
    IS = IS.to(device)
    return st1, st2, rewards, actions, dones, IS

def unpackBatchComplexObs(*dicts, device = DEVICE_DEFT):
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones, IS  = [], [], [], [], [], []
        pos1, pos2, tm1, tm2 = [], [], [], []
        for trajectory in dicts:
            states1 += [trajectory["st"]["frame"]]
            states2 += [trajectory["st1"]["frame"]]
            pos1 += [trajectory["st"]["position"]]
            pos2 += [trajectory["st1"]["position"]]
            tm1 += [trajectory["st"]["time"]]
            tm2 += [trajectory["st1"]["time"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
            IS += [trajectory["IS"]]
        st1 = Tcat(states1, dim=0)
        st2 = Tcat(states2, dim=0)
        pos1, pos2 = Tcat(pos1), Tcat(pos2)
        actions = Tcat(actions, dim=0)
        rewards = Tcat(rewards, dim=0)
        dones = Tcat(dones, dim=0)
        IS = Tcat(IS)
    else:
        trajectory = dicts[0]
        st1 = trajectory["st"]["frame"]
        pos1 = trajectory["st"]["position"]
        pos2 = trajectory["st1"]["position"]
        st2 = trajectory["st1"]["frame"]
        tm1 = trajectory["st"]["time"]
        tm2 = trajectory["st1"]["time"]
        actions = trajectory["action"]
        rewards = trajectory["reward"]
        dones = trajectory["done"]
        IS = trajectory["IS"]
    st1, st2 = st1.to(device), st2.to(device)
    pos1, pos2 = pos1.to(device).float(), pos2.to(device).float()
    tm1, tm2 = tm1.to(device).float(), tm2.to(device).float()
    st1 = {"frame":st1, "position":pos1, "time":tm1}
    st2 = {"frame":st2, "position":pos2, "time":tm2}
    rewards = rewards.to(device)
    dones = dones.to(device)
    actions = actions.to(device).unsqueeze(1).long()
    IS = IS.to(device)
    return st1, st2, rewards, actions, dones, IS