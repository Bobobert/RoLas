from rofl.utils.pg import Memory
from rofl.functions.const import *
from rofl.functions.dicts import mergeDicts
from rofl.functions.torch import A2T, L2T

class simpleMemory():
    """

        By default always looks for observation, reward, and 
        done as the most basic form of an experience. More keys
        can be tracked setting up within the configuration dict:'

            config->agent->memory_configuration->keys

        parameters
        ----------
        config: dict
        
    """
    _mem_ = None
    _keys_ = ["reward", "done"]
    _exp_ = None

    def __init__(self, config):
        self.config = config
        self.size = config["agent"].get("memory_size", 10**3)
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
        self._clean_()
            
    def _clean_(self):
        di = self._i_ - self._li_

        if di >= self.size:
            # For more than one.. but not this time
            #ds = di - self.size + 1
            temp = self._mem_[self._li_]
            self._mem_[self._li_] = None
            del temp
            self._li_ += 1

    def sample(self, size, device = DEVICE_DEFT):
        """
            Standard method to sample the memory. This is
            intended to be used as the main method to interface
            the memory.

            returns
            --------
            obsDict
        """
        if size > self._i_ - self._li_:
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
            n = random.randint(self._li_, self._i_ - 1)
            sample.append(self._mem_[n])
        
        return sample

    def processSample(self, sample, device):
        sample = mergeDicts(*sample, targetDevice = device)

        for k in self._keys_:
            aux = sample.get(k)
            if isinstance(aux, list):
                sample[k] = L2T(sample[k], device = device)

        return sample

class episodicMemory(simpleMemory):

    def __init__(self, config):
        super().__init__(config)
        self._keys_.append("return")

    def reset(self):
        super().reset()
        self._lastEpisode_ = 0

    def add(self, infoDict):
        super().add(infoDict)
        done = infoDict["done"]
        if done:
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

class dqnMemory(simpleMemory):
    def __init__(self, config):
        super().__init__(config)
        self.lhist = config["agent"]["lhist"]
        assert self.lhist > 0, "Lhist needs to be at least 1"
        self._keys_.append("rollout_return")

    def gatherSample(self, size):
        sample = []
        for _ in range(size):
            n = random.randint(self._li_ + self.lhist, self._i_ - 1)
            sample.append(self.getLhist(n))
        return sample
    
    def getLhist(self, i):
        op = item = self._mem_[i]
        obs = item["observation"]
        newObs = torch.zeros((1, self.lhist, *obs.shape[1:]), dtype = F_TDTYPE_DEFT)
        newObs[0,0] = obs.squeeze()
        for j in range(1, self.lhist):
            item = self._mem_[i - j]
            if item["done"]:
                break
            newObs[0, j] = item["observation"].squeeze()
        op["observation"] = newObs
        return op
