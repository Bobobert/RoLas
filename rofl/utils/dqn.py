from rofl.functions.const import *
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
                 action_dtype_in:np.dtype = np.uint8,
                 reward_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in

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
        self.t_buffer[self._i] = not t
        self._i = (self._i + 1) % self.capacity
        if self._i == 0:
            self.FO = True

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

    def sample(self, mini_batch_size:int, device = DEVICE_DEFT):
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
            ids = np.random.randint(self.LHist, self.capacity - 1 if self.FO else self._i - 2, 
                                    size=mini_batch_size)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = F_NDTYPE_DEFT)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i + 1, i - self.LHist, -1)):
                    s, _, _, t = self[j]
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
            terminals = torch.from_numpy(self.t_buffer[ids]).to(device).float()
            at = torch.from_numpy(self.a_buffer[ids]).to(device).long()
            rt = torch.from_numpy(self.r_buffer[ids]).to(device).float()
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
                 pos_dtype_in:np.dtype = I_NDTYPE_DEFT,
                 action_dtype_in:np.dtype = np.uint8,
                 reward_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in

        self.capacity = capacity
        self.LHist = LHist
        self.shapeHistOut = [LHist] + list(state_shape)
        self._i = 0
        self.FO = False

        self.s_buffer = np.zeros([capacity] + list(state_shape), dtype = state_dtype_in)
        self.p_buffer = np.zeros([capacity] + [2], dtype = pos_dtype_in)
        self.a_buffer = np.zeros(capacity, dtype = action_dtype_in)
        self.r_buffer = np.zeros(capacity, dtype = reward_dtype_in)
        self.t_buffer = np.ones(capacity, dtype = np.bool_) # Inverse logic

    def add(self, s, a, r, t):
        """
        Add one item
        """
        s, p = s["frame"], s["position"]
        self.s_buffer[self._i] = s
        self.p_buffer[self._i] = p
        self.a_buffer[self._i] = a
        self.r_buffer[self._i] = r
        self.t_buffer[self._i] = not t
        self._i = (self._i + 1) % self.capacity
        if self._i == 0:
            self.FO = True

    def __getitem__(self, i:int):
        if i < self._i or self.FO:
            i = i % self.capacity
            return (self.s_buffer[i],
                    self.p_buffer[i],
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
                0.0,
                False)

    def sample(self, mini_batch_size:int, device = DEVICE_DEFT):
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
            ids = np.random.randint(self.LHist, self.capacity - 1 if self.FO else self._i - 2, 
                                    size=mini_batch_size)
            st1 = np.zeros([mini_batch_size] + self.shapeHistOut, 
                           dtype = F_NDTYPE_DEFT)
            st2 = st1.copy()
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i + 1, i - self.LHist, -1)):
                    s, _, _, _, t = self[j]
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
            st1, st2 = {"frame":st1, "position":pos1}, {"frame":st2, "position":pos2}
            terminals = torch.from_numpy(self.t_buffer[ids]).to(device).float()
            at = torch.from_numpy(self.a_buffer[ids]).to(device).long()
            rt = torch.from_numpy(self.r_buffer[ids]).to(device).float()
            return {"st":st1,"st1":st2, "reward": rt, "action":at, "done":terminals, "obsType":"framePos"}
        else:
            raise IndexError("The memory does not contains enough transitions to generate the sample")

def unpackBatch(*dicts, device = DEVICE_DEFT):
    key = dicts[0].get("obsType", "standard")
    if key == "framePos":
        return unpackBatchComplexObs(*dicts, device = device)
    ### Standard Unpack
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones  = [], [], [], [], []
        for trajectory in dicts:
            states1 += [trajectory["st"]]
            states2 += [trajectory["st1"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
        st1 = Tcat(states1, dim=0)
        st2 = Tcat(states2, dim=0)
        actions = Tcat(actions, dim=0)
        rewards = Tcat(rewards, dim=0)
        dones = Tcat(dones, dim=0)

    else:
        trajectoryBatch = dicts[0]
        st1 = trajectoryBatch["st"]
        actions = trajectoryBatch["action"]
        rewards = trajectoryBatch["reward"]
        st2 = trajectoryBatch["st1"]
        dones = trajectoryBatch["done"]

    st1 = st1.to(device)
    st2 = st2.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    actions = actions.to(device).unsqueeze(1).long()
    return st1, st2, rewards, actions, dones

def unpackBatchComplexObs(*dicts, device = DEVICE_DEFT):
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones  = [], [], [], [], []
        pos1, pos2 = [], []
        for trajectory in dicts:
            states1 += [trajectory["st"]["frame"]]
            states2 += [trajectory["st1"]["frame"]]
            pos1 += [trajectory["st"]["position"]]
            pos2 += [trajectory["st1"]["position"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
        st1 = Tcat(states1, dim=0)
        st2 = Tcat(states2, dim=0)
        pos1, pos2 = Tcat(pos1), Tcat(pos2)
        actions = Tcat(actions, dim=0)
        rewards = Tcat(rewards, dim=0)
        dones = Tcat(dones, dim=0)

    else:
        trajectory = dicts[0]
        st1 = trajectory["st"]["frame"]
        pos1 = trajectory["st"]["position"]
        pos2 = trajectory["st1"]["position"]
        st2 = trajectory["st1"]["frame"]
        actions = trajectory["action"]
        rewards = trajectory["reward"]
        dones = trajectory["done"]

    st1, st2 = st1.to(device), st2.to(device)
    pos1, pos2 = pos1.to(device).float(), pos2.to(device).float()
    st1 = {"frame":st1, "position":pos1}
    st2 = {"frame":st2, "position":pos2}
    rewards = rewards.to(device)
    dones = dones.to(device)
    actions = actions.to(device).unsqueeze(1).long()
    return st1, st2, rewards, actions, dones