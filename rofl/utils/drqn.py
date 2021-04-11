from rofl.functions.const import *
from rofl.utils.dqn import MemoryReplay

def newZeroFromT(T):
    return T.new_zeros(T.shape).to(T.device)#.requires_grad_(T.requires_grad)

def newZero(batch, number, size, device):
    shape = [number, batch, size] 
    T = torch.zeros(shape, dtype = F_TDTYPE_DEFT).to(device)#.requires_grad_()
    return T

def hiddenShape(T):
    s = None
    if isinstance(T, tuple):
        # Asserting all the items have the same shape
        prev = None
        for t in T:
            if prev is not None and t.shape != prev:
                raise ValueError("Shapes on hidden state do not match!")
            if prev is None:
                prev = t.shape
        s = [len(T), *T[0].shape]
    elif isinstance(T, TENSOR):
        s = T.shape
    return tuple(s)

class hiddenState:
    hidden = None

    def initHidden(self, batch, number, size, device, lstm):
        if lstm:
            self.hidden = (newZero(batch,number,size,device),\
                            newZero(batch,number,size,device))
        else:
            self.hidden = newZero(batch,number,size,device)

    def reset(self):
        h = self.hidden
        if h is not None:
            if isinstance(h, tuple):
                newH = []
                for item in h:
                    newH += [newZeroFromT(item)]
                h = tuple(newH)
            elif isinstance(h, torch.Tensor):
                h = newZeroFromT(h) 
        self.hidden = h

    @property
    def shape(self):
        return hiddenShape(self.hidden)

class recurrentArguments:
    """
        Object design to pass and keep the hidden state of the
        recurrent units.
    """
    obs = None

    def __init__(self, config:dict):#RType:str, batch:int, number:int, size:int, units:int = 1):
        c = config["policy"]
        self.lstm = True if c["recurrent_unit"] is "lstm" else False
        units = assertIntPos(c["recurrent_units"])
        self.batch = assertIntPos(c["minibatch_size"])
        self.number  = assertIntPos(c["recurrent_layers"])
        self.size = assertIntPos(c["recurrent_hidden_size"])
        self.units = [hiddenState() for _ in range(units)]

    def __checkLen__(self, i:int):
        if i >= len(self.units):
            raise IndexError("This Argument holds just {} hidden states".format(len(self.units)))

    def __getitem__(self, i:int):
        self.__checkLen__(i)
        return self.units[i].hidden

    def __setitem__(self, i:int, x):
        self.__checkLen__(i)
        assert self.units[i].shape == hiddenShape(x), "Shapes must be equal"
        self.units[i].hidden = x

    def initHidden(self, batch = None, number=None, size=None, 
                        device = DEVICE_DEFT):
        b = self.batch if batch is None else assertIntPos(batch)
        n = self.number if number is None else assertIntPos(number)
        s = self.size if size is None else assertIntPos(size)
        for u in self.units:
            u.initHidden(b, n, s, device, self.lstm)

    def passHidden(self, *hiddens):
        assert len(hiddens) == len(self.units), "This cannot store more than its capacity"
        for n, hid in enumerate(hiddens):
            self.units[n].hidden = hid

    def reset(self,):
        for u in self.units:
            u.reset()
        self.obs = None
    
class MemoryReplayRecurrentFF(MemoryReplay):
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
                 recurrent_boot:int = RNN_BOOT_DEFT,
                 state_dtype_in:np.dtype = np.uint8,
                 pos_dtype_in:np.dtype = I_NDTYPE_DEFT,
                 action_dtype_in:np.dtype = np.uint8,
                 reward_dtype_in:np.dtype = F_NDTYPE_DEFT,
                 ):
        
        self.s_in_shape = state_shape
        self.s_dtype_in = state_dtype_in

        self.capacity = capacity
        self.rnnBoot = recurrent_boot
        self.shapeHistOut = list(state_shape)
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
        if self._i > mini_batch_size + 1 or self.FO:
            ids = np.random.randint(1, self.capacity - 1 if self.FO else self._i - 2, 
                                    size=mini_batch_size)
            bootS = np.zeros([self.rnnBoot, mini_batch_size] + self.shapeHistOut,
                            dtype = F_NDTYPE_DEFT)
            bootP = np.zeros([self.rnnBoot, mini_batch_size] + [2],
                            dtype = F_NDTYPE_DEFT)
            # make the boot
            for m, i in enumerate(ids):
                for n, j in enumerate(range(i - 1, i  - 1 - self.rnnBoot, -1)):
                    s, p, _, _, t = self[j]
                    if not t:
                        break
                    bootS[n][m] = s
                    bootP[n][m] = p
            # Passing to torch format
            st1 = torch.from_numpy(self.s_buffer[ids]).to(device).float().div(255).detach_().requires_grad_()
            st2 = torch.from_numpy(self.s_buffer[ids + 1]).to(device).float().div(255)
            pos1 = torch.from_numpy(self.p_buffer[ids]).to(device).float()
            pos2 = torch.from_numpy(self.p_buffer[ids + 1]).to(device).float()
            bootS = torch.from_numpy(bootS).to(device).div(255).detach_().requires_grad_()
            bootP = torch.from_numpy(bootP).to(device).float()
            st1, st2 = {"frame":st1, "position":pos1}, {"frame":st2, "position":pos2}
            boot = {"frame":bootS, "position":bootP}
            terminals = torch.from_numpy(self.t_buffer[ids]).to(device).float()
            at = torch.from_numpy(self.a_buffer[ids]).to(device).long()
            rt = torch.from_numpy(self.r_buffer[ids]).to(device).float()
            return {"st":st1,"st1":st2, "reward": rt, "action":at, "done":terminals, 
                    "obsType":"framePos", "recurrent_boot":boot}
        else:
            raise IndexError("The memory does not contains enough transitions to generate the sample")

def unpackBatch(*dicts, device = DEVICE_DEFT):
    if len(dicts) > 1:
        states1, states2, actions, rewards, dones  = [], [], [], [], []
        pos1, pos2 = [], []
        bootS, bootP = [],[]
        for trajectory in dicts:
            states1 += [trajectory["st"]["frame"]]
            states2 += [trajectory["st1"]["frame"]]
            pos1 += [trajectory["st"]["position"]]
            pos2 += [trajectory["st1"]["position"]]
            actions += [trajectory["action"]]
            rewards += [trajectory["reward"]]
            dones += [trajectory["done"]]
            bootS += [trajectory["recurrent_boot"]["frame"]]
            bootP += [trajectory["recurrent_boot"]["position"]]
        st1 = Tcat(states1, dim=0)
        st2 = Tcat(states2, dim=0)
        pos1, pos2 = Tcat(pos1), Tcat(pos2)
        actions = Tcat(actions, dim=0)
        rewards = Tcat(rewards, dim=0)
        dones = Tcat(dones, dim=0)
        bootS, bootP = Tcat(bootS), Tcat(bootP)
    else:
        trajectory = dicts[0]
        st1 = trajectory["st"]["frame"]
        pos1 = trajectory["st"]["position"]
        pos2 = trajectory["st1"]["position"]
        st2 = trajectory["st1"]["frame"]
        actions = trajectory["action"]
        rewards = trajectory["reward"]
        dones = trajectory["done"]
        bootS = trajectory["recurrent_boot"]["frame"]
        bootP = trajectory["recurrent_boot"]["position"]

    st1, st2 = st1.to(device), st2.to(device)
    pos1, pos2 = pos1.to(device).float(), pos2.to(device).float()
    st1 = {"frame":st1, "position":pos1}
    st2 = {"frame":st2, "position":pos2}
    bootS, bootP = bootS.to(device), bootP.to(device).float()
    boot = {"frame":bootS, "position":bootP}
    rewards = rewards.to(device)
    dones = dones.to(device)
    actions = actions.to(device).unsqueeze(1).long()
    return st1, st2, rewards, actions, dones, boot
