from rofl.functions.const import *
from rofl.functions.functions import rnd, newZero
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
    _exp_ = None
    memType = 'simple'
    __keysDeft__ = [("reward", F_TDTYPE_DEFT), ("done", B_TDTYPE_DEFT)]

    def __init__(self, config, *aditionalKeys):
        self.size = config["agent"].get("memory_size", DEFT_MEMORY_SIZE)
        self.gamma = config["agent"]["gamma"]
        self.gae = config["agent"]["gae"]
        self.lmbda = config["agent"]["lambda"]
        self._mem_, self._i_, self._li_ = None, 0, 0
        self.fillOnce = False
        self._keys_ = self.__keysDeft__.copy()
        for key in aditionalKeys:
            self._keys_.append(key)

    def reset(self,):
        self._mem_, self._i_, self._li_ = [None]*self.size, 0, 0
        self.fillOnce = False
        return self
    
    def add(self, infoDict):
        self._mem_[self._i_] = infoDict
        self._i_ = self._i_ + 1 % self.size
        if self._i_ == 0:
            self.fillOnce = True

    @property
    def last(self,):
        return self._i_ - 1
    
    def sample(self, size, device = DEVICE_DEFT, keys = None):
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
        return self.createSample(self.gatherSample(size), device, keys)

    def gatherMem(self):
        """
            Returns all the items from memory

            returns
            -------
            index generator
        """
        if self.fillOnce:
            return itemsSeq(0, self.size)
        return itemsSeq(0, self._i_)

    def gatherSample(self, size):
        """
            Gathers the indexes from memory for the sample
            generation.

            returns
            -------
            index generator
        """
        if self.fillOnce:
            return itemsRnd(0, self.size - 1, size)
        return itemsRnd(0, self.last, size)

    def createSample(self, genSample, device, keys) -> dict:
        """
            Generates and process the sample from the gatherSample method. 
            This could be done per item or in bulk. Either way is expected to
            return a single obsDict.

            returns
            --------
            obsDict
        """
        sample = mergeDicts(*[self[i] for i in genSample], targetDevice = device, keys = keys)

        for key, dtype in self._keys_:
            aux = sample.get(key)
            if aux is None: # When the key in memory was not asked in keys for the merge result
                continue
            if isinstance(aux, TENSOR):
                continue
            elif isinstance(aux, list):
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
        self._assertSize_(memory)
        self._mem_[:len(memory)] = memory._mem_

    def addMemory(self, memory):
        """
            Add the experiences from a memory to another. 
        """
        self._assertSize_(memory)
        lTarget = len(memory)
        if lTarget + self._i_ > self.size:
            underThat = self.size - self._i_
            overThis = lTarget + self._i_ % self.size
            self._mem_[self._i_:] = memory._mem_[:underThat]
            self._mem_[:overThis] = memory._mem_[underThat:]
            self._i_, self.fillOnce = overThis, True
        else:
            self._mem_[self._i_:self._i_ + lTarget] = memory._mem_
            self._i_ = self._i_ + lTarget % self.size
            if self._i_ == 0:
                self.fillOnce = True

    def _assertSize_(self, memory):
        lTarget = len(memory)
        if lTarget > self.size:
            raise ValueError('That memory (%d) is bigger than this (%d)' % (lTarget, len(self)))

    def __len__(self):
        if self.fillOnce:
            return self.size
        return self._i_

    def __getitem__(self, i):
        return self._mem_[i]

    def __repr__(self) -> str:
        s = 'Memory %s with %d capacity, %d items stored'%(self.memType, self.size, len(self))
        return s

class episodicMemory(simpleMemory):
    """
        Meant to store and process one episode at a time
    """
    memType = 'episodic simple'

    def __init__(self, config, *additionalKeys):
        keys = [("return", F_TDTYPE_DEFT)]
        super().__init__(config, *keys, *additionalKeys)

    def reset(self):
        super().reset()
        self._lastEpisode_ = -1

    def add(self, infoDict):
        super().add(infoDict)
        if infoDict["done"]:
            self.resolveReturns()

    def resolveReturns(self):
        lastEpsisode = self._lastEpisode_
        self._lastEpisode_ = last = self.last
        gamma = self.gamma

        lastReturn = self[last].get('bootstrapping', 0.0)
        if isinstance(lastReturn, TENSOR):
            lastReturn = lastReturn.cpu().item()
        for i in range(last, lastEpsisode, - 1):
            lastDict = self[i]
            lastReturn = lastDict['return'] = lastDict["reward"] + gamma * lastReturn 

    def getEpisode(self, device = DEVICE_DEFT, keys = None):
        if self._lastEpisode_ == -1:
            raise AttributeError("Memory does not have an episode ready!")
        return self.createSample(self.gatherMem(), device, keys)

class multiMemory:
    name = 'multi episodic'

    def __init__(self, config, *additionalKeys):
        self.n = config['agent']['workers']
        self.config = config
        self._addKeys = additionalKeys
        self._hasInit = False
        self._memories, self._memList = {}, []

    def reset(self):
        for mem in self._memList:
            mem.reset()
    
    def __getitem__(self, i):
        memories = self._memories
        mem = memories.get(i)
        if mem is not None:
            return mem
        elif mem is None and len(memories) < self.n:
            new = episodicMemory(self.config, *self._addKeys)
            new.reset()
            self._memList.append(new)
            memories[i] = new
            return new
        elif len(memories) >= self.n:
            raise ValueError('New memories cannot be crated, already at max capacity.')

    def __repr__(self) -> str:
        s = 'Memory %s, managing %d units.' % (self.name, len(self._memories))
        return s

    def add(self, *obsDict):
        for dict_ in obsDict:
            iD = dict_['id']
            mem = self[iD]
            mem.add(dict_)

    def getEpisodes(self, device = DEVICE_DEFT, keys = None, forceResolve = True):
        episodes = []
        for mem in self._memList:
            if forceResolve: mem.resolveReturns()
            episode = mem.getEpisode(device, keys)
            episodes.append(episode)
        return episodes

    def getSamples(self, size, device = DEVICE_DEFT, keys = None):
        samples = []
        for mem in self._memList:
            sample = mem.getSample(size, device, keys)
            samples.append(sample)
        return samples
            
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
        if self.fillOnce:
            return itemsRnd(0, self.size - 1, size)
        return itemsRnd(1, self._i_ - 1, size)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        if item.get('frame') is None:
            prevItem = super().__getitem__(i-1)
            if prevItem.get('done', True): # if gatherMem is called the first item will have to have frame in zeros!
                item['frame'] = self.zeroFrame
            else:
                item['frame'] = prevItem['next_frame']
        return item

    def createSample(self, genSample, device, keys):
        sample = super().createSample(genSample, device, keys)
        sample['observation'] = array2Tensor(sample['frame'], device, batch=True).div(255)
        sample['next_observation'] = array2Tensor(sample['next_frame'], device, batch=True).div(255)
        sample['action'] = list2Tensor(sample['action'], device, torch.int64)
        return sample
