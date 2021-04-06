import os
import sys
import time
from torch import save, load, device
import pickle

LIMIT_4G = 3.8 * 1024 ** 3

def timeFormated() -> str:
    return time.strftime("%H-%M_%d-%b-%y", time.gmtime())

def timeFormatedS() -> str:
    return time.strftime("%H-%M-%S_%d-%b-%y", time.gmtime())

def timeDiffFormated(start):
    tock = time.time()
    total = tock - start
    hours = int(total//3600)
    mins = int(total%3600//60)
    secs = int(total%3600%60//1)
    if hours > 0:
        s = "{}h {}m {}s".format(hours, mins, secs)
    elif mins > 0:
        s = "{}m {}s".format(mins, secs)
    elif secs > 0:
        s = "{}s".format(secs)
    else:
        s = "< 0s"
    return s, tock

def goToDir(path):
    home = os.getenv('HOME')
    try:
        os.chdir(os.path.join(home, path))
    except:
        os.chdir(home)
        os.makedirs(path)
        os.chdir(path)
    return os.getcwd()

def createFolder(path:str, mod:str):
    start = mod + '_' + timeFormated()
    new_dir = os.path.join(path, start)
    new_dir = goToDir(new_dir)
    return start, new_dir

def timeToStop(results, expected = None):
    tock = time.time()
    diff = tock - results["time_start"]
    results["time_elapsed"] = diff
    results["time_execution"] += [timeFormatedS()]
    stop = False
    if ((diff // 60) >= expected) and (expected is not None):
        stop = True
    return results, stop

class Tocker:
    def __init__(self):
        self.tick
    @property
    def tick(self):
        self.start = time.time()
        return self.start
    @property
    def tock(self):
        s, self.start = timeDiffFormated(self.start)
        return s
    @property
    def tocktock(self):
        """
        Returns the time elapsed since the last tick in minutes
        """
        return (time.time() - self.start) * 0.016666667
    def lockHz(self, Hz:int):
        tHz = 1 / Hz
        remaind = time.time() - self.start
        remaind = tHz - remaind
        if remaind > 0:
            time.sleep(remaind)
            return True

class Stack:
    """
    Dict stack working in a FIFO manner
    """
    def __init__(self):
        self.stack = dict()
        self.min = 0
        self.actual = 0
    def add(self, obj):
        self.stack[self.actual] = obj
        self.actual += 1
    def pop(self):
        poped = self.stack[self.min]
        self.stack.pop(self.min)
        self.min += 1
        return poped
    def __len__(self):
        return len(self.stack)

class Reference:
    def __init__(self, obj, 
                        name: str,
                        limit:int,
                        torchType:bool = False,
                        device = device("cpu")):
        self.torchType = torchType
        self.name = name
        self.ref = obj
        self.prevVersions = Stack()
        self.limit = limit
        self.device = device
        self._version = 0
    
    def save(self, path):
        if self.torchType:
            self.saveTorch(path)
        else:
            self.savePy(path)
        self.clean(path)

    def clean(self, path):
        if len(self.prevVersions) >= self.limit:
            target = self.prevVersions.pop()
            #target = os.path.join(path, target)
            os.remove(target)
    
    @staticmethod
    def loaderAssist(path):
        os.chdir(path)
        files = os.listdir()
        print("Files on direction:")
        for n, File in enumerate(files):
            print("{} : {}".format(n, File))
        while 1:
            choice = input("Enter the number for the file to load :")
            choice = int(choice)
            if choice > len(files) or not isinstance(choice, int) or choice < 0:
                print("Number not valid. Please try again.")
            else:
                break
        return os.path.join(path, files[choice])

    def load(self, path):
        print("Trying to load in object {}".format(self.name))
        target = self.loaderAssist(path)
        if self.torchType:
            self.loadTorch(target, self.device)
        else:
            self.loadObj(target)
    
    def loadTorch(self, path, device):
        model = load(path, map_location=device)
        self.ref.load_state_dict(model, strict = True)
        print("Model successfully loaded from ", path)
        
    def loadObj(self, path):
        fileHandler = open(path, 'rb')
        self.ref = pickle.load(fileHandler)
        fileHandler.close()
        print("Object successfully loaded from ", path)

    def saveTorch(self, path):
        name = self._gen_name() + ".modelst"
        path = os.path.join(path, name)
        try:
            stateDict = self.ref.state_dict()
            save(stateDict, path)
            self.prevVersions.add(path)
        except:
            None

    def savePy(self, path):
        name = self._gen_name() + ".pyobj"
        path = os.path.join(path, name)
        if sys.getsizeof(self.ref) < LIMIT_4G:
            fileHandler = open(path, "wb")
            pickle.dump(self.ref, fileHandler)
            fileHandler.close()
            self.prevVersions.add(path)

    def _gen_name(self):
        self._version += 1
        return self.name + "_v{}".format(self._version) + "_" + timeFormated()

class Saver():
    """
    Object that administrates objects to dump
    save files if possible.

    parameters
    ----------
    envName: str

    path: str
        Path relative to Home to dump the saved files
    """
    def __init__(self, envName:str, 
                    path:str = "PG_results/",
                    limitTimes:int = 5,
                    saveFreq:int = 40):
        
        self.startPath, self.dir = createFolder(path, envName)
        self._objRefs_ = []
        self.names = set()
        self.limit = limitTimes
        self.time = Tocker()
        self.freq = saveFreq

    def start(self):
        self.time.tick

    def check(self):
        if self.time.tocktock >= self.freq:
            self.saveAll()
            self.time.tick

    def addObj(self, obj, 
                objName:str,
                isTorch:bool = False,
                device = device("cpu")):

        if objName in self.names:
            raise KeyError
        self.names.add(objName)
        self._objRefs_ += [Reference(obj, 
                                    objName, 
                                    self.limit,
                                    isTorch,
                                    device)]
    
    def saveAll(self):
        for ref in self._objRefs_:
            ref.save(self.dir)

    def load(self, path):
        for ref in self._objRefs_:
            ref.load(path)