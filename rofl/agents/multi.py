from .base import Agent
from rofl.functions.const import *
from rofl.functions.dicts import mergeDicts
import ray

READY = 1
WORKING = 2
DONE = 3

class agentMaster():
    reseted = False
    def __init__(self, config, policy, envMaker, agentClass,
                    tbw = None, **kwargs):

        self.mainAgent = agentClass(config, policy, envMaker, **kwargs)
        self.policy = policy
        self.config = config

        nags = config["agent"].get("workers", NCPUS)
        self._nAgents = nags = NCPUS if nags > NCPUS or nags < 1 else nags

        ray.init(num_cpus = nags)
        ragnt = ray.remote(agentClass)

        self.workers = workers = dict()
        s1, s2 = config["agent"].get("seedTrain", TRAIN_SEED), config["agent"].get("seedTest", TEST_SEED)
        for i in range(nags):
            nconfig, worker = config.copy(), Worker()
            workers[i] = worker
            nconfig["agent"]["id"] = worker.id = i
            nconfig["env"]["seedTrain"] = s1 + i + 1
            nconfig["env"]["seedTest"] = s2 + i + 1
            worker.worker = ragnt.remote(nconfig, None, envMaker)
    
    def reset(self):
        for agent in self.agents.values():
            agent.reset.remote()
        return self.resolveAll()

    def envStep(self, actions, ids):
        # distribute actions
        for i, Id in enumerate(ids):
            worker = self.workers[Id]
            action = actions[i]
            worker.ref = worker().envStep.remote(action)
        # resolve workers
        results = self.resolveAll()
        # merge infoDicts
        return mergeDicts(*results, targetDevice = self.policy.device)

    def fullStep(self):
        actions, ids = self.policy.getActions(self.lastObs)
        self.lastObs = obs = self.envStep(actions, ids)
        return obs

    def getBatch(self, size: int, proportion: float = 1.0):
        pass
    
    def resolveAll(self):
        """
            Working in a synchronous manner. Resolver each worker
            with a pending task
        """
        for w in self.workers.values():
            w.resolve()
        return [w.result for w in self.workers.values() if w.status == DONE]

        working, ids = [], []
        for w in self.workers.values():
            working.append(w.objRef)
            ids.append(w.id)
        results = ray.get(working)
        for i, r in enumerate(results):
            pass
        # TODO complete ?


class Worker:
    """
        Design to store and manage the ray actors

        Status codes:
        0 - not initialized
        1 - initialized / ready
        2 - working
        3 - done
    """
    worker = None
    _objRef = None
    id = None
    _result, _newconsult = None, False

    @property
    def status(self):
        if self.worker is None:
            return 0
        elif self.ref is None:
            if self._result is None:
                return READY
            else:
                return DONE
        elif not self._newconsult:
            return WORKING
        else:
            print("Eh? Who am I?")
    
    @property
    def ref(self):
        return self._objRef

    @ref.setter
    def ref(self, r):
        self._newconsult = False
        self._objRef = r

    @property
    def result(self):
        if self.status == WORKING:
            print("Warning: Worker {} haven't been resolved".format(self.id))
            return None
        else:
            temp = self._result
            self._result = None
            return temp
    
    @result.setter
    def result(self, x):
        self._objRef = None
        self._newconsult = True
        self._result = x

    def resolve(self, timeout = None):
        """
            parameters
            ----------
            timeout: int
                in miliseconds, the time to wait the worker
                to finish it's process

            Warning
            -------
            This method will stall the main process until its 
            worker is ready if timeout is None
        """
        if self.status != WORKING:
            pass

        if timeout is not None:
            timeout = timeout / 1000

        try:
            self.result = ray.get(self.ref, timeout = timeout)
        except:
            pass

    def __call__(self):
        return self.worker

class workerManager:
    None
