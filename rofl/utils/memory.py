from rofl.functions.const import *
from rofl.functions.functions import rnd
from rofl.functions.dicts import mergeDicts
from rofl.functions.torch import array2Tensor, list2Tensor

def itemsSeq(low, high):
    """
        Generator for the range [low, high)
    """
    for i in range(low, high):
        yield i

def itemsRnd(low, high, nItems):
    """
        Generator for nItems in the interval [low, high]
    """
    for i in range(nItems):
        yield rnd.randint(low, high)

class simpleMemory():
    """

        By default always looks for observation, reward, and 
        done as the most basic form of an experience.

        parameters
        ----------
        - config: dict

        - aditionalKeys: tuple(str, Any dtype)
            Specify what aditional keys are required to process after
            gathering a sample to a torch.tensor and its data type.
            eg. ('return', torch.float16)
        
    """
    _exp_, _slack_ = None, 4
    memType = 'simple'
    __keysDeft__ = [("reward", F_TDTYPE_DEFT), ("done", B_TDTYPE_DEFT)]

    def __init__(self, config, *aditionalKeys):
        self.size = config["agent"].get("memory_size", DEFT_MEMORY_SIZE)
        self._slack_ *= self.size
        self.gamma = config["agent"]["gamma"]
        self.gae = config["agent"]["gae"]
        self.lmbda = config["agent"]["lambda"]
        self._mem_, self._i_, self._li_ = None, 0, 0
        self._keys_ = self.__keysDeft__.copy()
        for key in aditionalKeys:
            self._keys_.append(key)

    def reset(self,):
        self._mem_, self._i_, self._li_ = [], 0, 0
        return self
    
    def add(self, infoDict):
        self._mem_.append(infoDict)
        self._i_ += 1
        self._cleanItem_()
    
    @property
    def diff(self,):
        return self._i_ - self._li_

    @property
    def last(self,):
        return self._i_ - 1

    def _cleanItem_(self):
        if self.diff >= self.size:
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
        memSize = len(self)
        if size > memSize:
            raise ValueError("Not enough data to generate sample")
        if size == memSize or size < 0:
            return self.createSample(self.gatherMem(), device)
        return self.createSample(self.gatherSample(size), device)

    def gatherMem(self):
        """
            Returns all the items from memory

            returns
            -------
            index generator
        """
        return itemsSeq(self._li_, self._i_)

    def gatherSample(self, size):
        """
            Gathers the indexes from memory for the sample
            generation.

            returns
            -------
            index generator
        """
        return itemsRnd(self._li_, self._i_ - 1, size)

    def createSample(self, genSample, device):
        """
            Generates and process the sample from the gatherSample method. 
            This could be done per item or in bulk. Either way is expected to
            return a single obsDict.

            returns
            --------
            obsDict
        """
        sample = mergeDicts(*[self[i] for i in genSample], targetDevice = device)

        for key, dtype in self._keys_:
            aux = sample.get(key)
            if isinstance(aux, list):
                sample[key] = list2Tensor(aux, device, dtype)
            elif isinstance(aux, ARRAY):
                sample[key] = array2Tensor(aux, device, dtype, batch=True)
            elif isinstance(aux, (int, float)): # when sample[N] = 1, then some elements could raise expception
                sample[key] = list2Tensor([aux], device, dtype)
            else:
                raise NotImplementedError('This wasnt expected yet... oopsy')

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

    def __len__(self):
        return self.diff

    def __getitem__(self, i):
        if i >= self._i_ or i < self._li_:
            return dict()
        return self._mem_[i]

    def __repr__(self) -> str:
        s = 'Memory %s with %d capacity, %d items stored'%(self.memType, self.size, self.diff)
        return s

class episodicMemory(simpleMemory):
    """
        Meant to store and process one episode at a time
    """
    memType = 'episodic simple'

    def __init__(self, config, *additionalKeys):
        super().__init__(config, ("return", F_TDTYPE_DEFT), *additionalKeys)

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
        # this could be done better in here? Anyway an iterator throu all the dicts is needed anyway
        # using dicts I don-t see a better way.
        lastReturn = self[self.last].get('bootstrapping', 0.0)
        for i in range(self.last, self._lastEpisode_ - 1, - 1):
            lastDict = self[i]
            lastReturn = lastDict['return'] = lastDict["reward"] + self.gamma * lastReturn 

        self._lastEpisode_ = self._i_

    def getEpisode(self, device = DEVICE_DEFT):
        if self._lastEpisode_ == 0:
            raise AttributeError("Memory does not have an episode ready!")
        return self.createSample(self.gatherMem(), device)

class dqnMemory(simpleMemory):
    memType = 'dqn v0'

    def __init__(self, config):
        super().__init__(config)
        self.lhist = config["agent"]["lhist"]
        assert self.lhist > 0, "Lhist needs to be at least 1"
        from rofl.utils.dqn import genFrameStack
        self.zeroFrame = genFrameStack(config)
    
    @staticmethod
    def lHistMem(memory, i, lHist): #Not in use, saving all the frames in this version :c
        item = memory[i]
        obs = item["observation"]
        newObs = torch.zeros((1, lHist, *obs.shape[1:]), dtype = F_TDTYPE_DEFT)
        newObs[0,0] = obs.squeeze()
        for j in range(1, lHist):
            item = memory[i - j]
            if item["done"]:
                break
            newObs[0, j] = item["observation"].squeeze()
        item["observation"] = newObs
        return item

    def gatherSample(self, size):
        return itemsRnd(self._li_ + 1, self._i_ - 1, size)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        if item.get('frame') is None:
            prevItem = super().__getitem__(i-1)
            if prevItem.get('done', True): # if gatherMem is called the first item will have to have frame in zeros!
                item['frame'] = self.zeroFrame# this should keep only references
            else:
                item['frame'] = prevItem['next_frame']
        return item

    def createSample(self, genSample, device):
        sample = super().createSample(genSample, device)
        sample['observation'] = array2Tensor(sample['frame'], device, batch=True).div(255)
        sample['next_observation'] = array2Tensor(sample['next_frame'], device, batch=True).div(255)
        sample['action'] = list2Tensor(sample['action'], device, torch.int64)
        return sample
