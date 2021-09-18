from rofl.functions.const import *
from rofl.functions.dicts import mergeDicts
from rofl.functions.torch import array2Tensor, list2Tensor

class simpleMemory():
    """

        By default always looks for observation, reward, and 
        done as the most basic form of an experience.

        parameters
        ----------
        config: dict
        
    """
    _mem_ = None
    _keys_ = ["reward", "done"]
    _exp_, _slack_ = None, 10

    def __init__(self, config):
        self.config = config
        self.size = config["agent"].get("memory_size", 10**3)
        self._slack_ *= self.size
        self.gamma = config["agent"]["gamma"]
        self.gae = config["agent"]["gae"]
        self.lmbda = config["agent"]["lambda"]

    def reset(self,):
        self._mem_, self._i_ = [], 0
        self._li_ = 0

        return self
    
    def add(self, infoDict):
        self._mem_.append(infoDict)
        self._i_ += 1
        self._cleanItem_()
    
    @property
    def diff(self,):
        return self._i_ - self._li_

    def _cleanItem_(self):
        diff = self.diff

        if diff >= self.size:
            # For more than one.. but not this time
            #ds = di - self.size + 1
            self._mem_[self._li_] = None
            self._li_ += 1
        if self._li_ > self._slack_:
            self._cleanMem_()
    
    def sample(self, size, device = DEVICE_DEFT):
        """
            Standard method to sample the memory. This is
            intended to be used as the main method to interface
            the memory.

            returns
            --------
            obsDict
        """
        if size > self.diff:
            raise ValueError("Not enough data to generate sample")
    
        return self.processSample(self.gatherSample(size), device)

    def gatherSample(self, size):
        """
            Gathers the items from the memory to be processed

            returns
            -------
            list of obsDict
        """
        sample = []
        for _ in range(size):
            n = rnd.randint(self._li_, self._i_ - 1)
            sample.append(self._mem_[n])
        
        return sample

    def processSample(self, sample, device):
        """
            Process the sample from the gathe sample method. It should
            the this item per item or in bulk. Either way is expected to
            return a single obsDict.

            returns
            --------
            obsDict
        """
        sample = mergeDicts(*sample, targetDevice = device)

        for k in self._keys_:
            aux = sample.get(k)
            if isinstance(aux, list):
                sample[k] = list2Tensor(sample[k], device = device)

        return sample
    
    def copyMemory(self, memory):
        """
            Resets and copy the takes the target memory reference to
            the memory list. This does not copy any object howsoever.
            Modifies the memory state to match the target withour changing
            the memory configuration.
        """
        self.reset()
        self._mem_ = memory._mem_
        self._i_, self._li_ = memory._i_, memory._li_
        if memory.diff > self.size:
            self._li_ = memory._i_ - self.size + 1 #TODO: check this
    
    def _cleanMem_(self):
        self._mem_ = self._mem_[max(0, self._i_ - self.size, self._li_):self._i_]
        self._i_, self._li_ = len(self._mem_), 0

    def addMemory(self, memory):
        """
            Add the experiences from a memory to another. 
        """
        pass
        self._mem_ += memory._mem_
        self._i_ += memory.diff
        if self.diff > self.size:
            self._cleanMem_()

class episodicMemory(simpleMemory):

    def __init__(self, config):
        super().__init__(config)
        self._keys_.append("return")

    def reset(self):
        super().reset()
        self._lastEpisode_ = 0

    def add(self, infoDict):
        super().add(infoDict) 
        if infoDict["done"]:
            self.resolveReturns()

    def resolveReturns(self):
        if self._lastEpisode_ < self._li_:
            self._lastEpisode_ = self._li_
        
        # Collect the episode rewards
        lastReturn = 0.0
        for i in range(self._i_ - 1, self._lastEpisode_ - 1, -1):
            lastDict = self._mem_[i]
            lastReturn = lastReturn * self.gamma + lastDict["reward"]
            lastDict["return"] = lastReturn

        self._lastEpisode_ = self._i_ - 1

    def getEpisode(self, device = DEVICE_DEFT):
        return self.processSample(self._mem_, device)

def generateLHist(memory, i, lHist):
    item = memory._mem_[i]
    obs = item["observation"]
    newObs = torch.zeros((1, lHist, *obs.shape[1:]), dtype = F_TDTYPE_DEFT)
    newObs[0,0] = obs.squeeze()
    for j in range(1, lHist):
        item = memory._mem_[i - j]
        if item["done"]:
            break
        newObs[0, j] = item["observation"].squeeze()
    item["observation"] = newObs
    return item

class dqnMemory(simpleMemory):
    def __init__(self, config):
        super().__init__(config)
        self.lhist = config["agent"]["lhist"]
        assert self.lhist > 0, "Lhist needs to be at least 1"
        self._keys_.append("rollout_return")

    def gatherSample(self, size):
        sample = []
        for _ in range(size):
            i = rnd.randint(self._li_ + self.lhist, self._i_ - 1)
            sample.append(self.getLhist(i))
        return sample
    
    def getLhist(self, i):
        return generateLHist(self, i, self.lhist)
