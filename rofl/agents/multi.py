from .base import Agent
from rofl.functions.const import *
from rofl.functions.dicts import mergeDicts
import ray

# Status Flags
READY = 1
WORKING = 2
DONE = 3

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
            return False

        if timeout is not None:
            timeout = timeout / 1000

        try:
            self.result = ray.get(self.ref, timeout = timeout)
            return True
        except:
            return False

    def __call__(self):
        return self.worker

class agentMaster():
    
    def __init__(self, config, policy, envMaker, agentClass, **kwargs):

        #self.mainAgent = agentClass(config, policy, envMaker, **kwargs)
        self.policy = policy
        self.config = config

        nags = config["agent"].get("workers", NCPUS)
        nags += 1 if nags == 1 else 0
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
            worker.worker = ragnt.remote(nconfig, policy.new() if policy is not None else None, envMaker)
    
    @property
    def device(self):
        return self.policy.device

    def reset(self):
        for worker in self.workers.values():
            worker.ref = worker().reset.remote()
        results = self.syncResolve()
        self.lastObs = mergeDicts(*results, targetDevice = self.device)
        return results

    def envStep(self, actions, ids):
        # distribute actions
        for i, Id in enumerate(ids):
            worker = self.workers[Id]
            action = actions[i]
            worker.ref = worker().envStep.remote(action)
        # resolve workers
        results = self.syncResolve()
        # merge infoDicts
        return mergeDicts(*results, targetDevice = self.device)

    def fullStep(self):
        actions, ids = self.policy.getActions(self.lastObs)
        self.lastObs = obs = self.envStep(actions, ids)
        return obs

    def getBatch(self, size: int, proportion: float = 1.0):
        pass

    def listWorkers(self, timeout = 0):
        """
            From the actors returns the first with
            a READY status.

            Ignores the agent with DONE status, as they remain
            to be collected.

            arameters
            ----------
            timeout: float
                If greater than 0, is the Time to wait 
                to resolve an agent if is working status
                in a only an attemp.

            returns
            -------
            list of workers with ready status, list of workers with working
            status, list of workers with done status
        """
        working, ready, done = [], [], []
        for w in self.workers.values():
            s = w.states
            if s == READY:
                ready.append(w)
            elif s == WORKING:
                # try to resolve
                if timeout > 0:
                    if w.resolve(timeout):
                        done.append(w)
                working.append(w)
            elif s == DONE:
                done.append(w)
        return ready, working, done

    def syncResolve(self, working = None):
        """
            Working in a synchronous manner. Resolver each worker
            with a pending task.

            parameters
            -----------
            working: list of workers or NoneType
                Can resolve from a given list of agents or pass
                a NoneType to apply to all the workers from this
                object.

            returns
            -------
            list of results
        """
        if working == None:
            working = [w for w in self.workers.values() if w.status == WORKING]
        for w in working:
            w.resolve()
        return [w.result for w in working] 

    def asyncResolve(self, working = None, n = 1):
        """
            parameters
            ----------
            working: list or None
                if a list of workers is given solves
                from there. Else generates a list.

            n: int
                Number of workers to be solved in this
                pass if able. At least always will return
                one solved worker

            returns
            -------
            list of results, list of workers freed, and list of
            workers with reamaining working status
        """
        if working is None:
            _, working, _ = self.listWorkers()
        if len(working) < 1:
            print("Warning: asyncResolve has no working workers")
            return [], []
        n = n if n <= len(working) else len(working)
        readyIDS, workingIDs = ray.wait([w.ref for w in working], num_returns=n)
        readyIDS, results, solved, stillWork = set(readyIDS), [], [], []
        for w in working:
            if w.ref in readyIDS:
                w.resolve()
                results.append(w.result)
                solved.append(w)
            else:
                stillWork.append(w)
        return results, solved, stillWork

    def close(self):
        for w in self.workers.values():
            del w.worker
        ray.shutdown()

class agentSync(agentMaster):
    
    @property
    def device(self):
        return self.policy.device

    def reset(self):
        for worker in self.workers.values():
            worker.ref = worker().reset.remote()
        results = self.syncResolve()
        self.lastObs = mergeDicts(*results, targetDevice = self.device)
        return results

    def fullStep(self):
        for w in self.workers.values():
            w.ref = w().fullStep.remote()
        
        return mergeDicts(*self.syncResolve, targetDevice = self.device)

    def getEpisodes(self):
        for w in self.workers.values():
            w.ref = w().getEpisode.remote()

        return mergeDicts(*self.syncResolve())

    def sync(self, batch):
        self.policy.update(batch)

        state = self.policy.currentState()
        for w in self.workers.values():
            w.ref = w().loadState.remote(state)
        
        self.syncResolve()