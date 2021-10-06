from rofl.config.config import createPolicy
from rofl.functions.torch import cloneNet, newNet
from .base import Agent
from rofl.functions.const import *
from rofl.functions.functions import ceil
from rofl.functions.dicts import mergeDicts, mergeResults
import ray

# Status Flags
READY = 1
WORKING = 2
DONE = 3

class Worker:
    _eqvDict = {
        READY: 'ready',
        WORKING: 'working',
        DONE: 'done',
    }
    """
        Design to store and manage the ray actors

        Status codes:
        1 - initialized / ready
        2 - working
        3 - done
    """
    def __init__(self, rayWorker, id: int) -> None:
        self.worker = rayWorker
        self.id = id
        self._objRef = None
        self._result, self._newconsult = None, False

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
            raise Exception("Worker %d haven't been resolved" % self.id)
            return
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
    
    def __repr__(self) -> str:
        s = 'Worker for Ray. ID: %d. Status: %s' %(self.id, self.status)
        return s

class agentMaster():
    name = 'Agent Master v0'

    def __init__(self, config, policy, envMaker, **kwargs):

        self.policy = policy
        self.config = config

        self.tbw = kwargs.get('tbw')
        self.testCalls = 0

        nWorkers = config["agent"].get("workers", NCPUS)
        nWorkers += 1 if nWorkers == 1 else 0
        self._nWorkers = nWorkers = NCPUS if nWorkers > NCPUS or nWorkers < 1 else nWorkers
        self.actorIsShared = actorShared = config['policy']['shared_memory']

        import rofl.agents as agents
        agentClass = getattr(agents, config['agent']['workerClass']) # should raise error when ill config
        ray.init(num_cpus = nWorkers)
        ragnt = ray.remote(agentClass)

        workerPolicy = None
        nActor = actor = policy.actor
        nBl = baseline = getattr(policy, 'baseline') if not getattr(policy, 'actorHasCritic', True) else None
        if actorShared: actor.shareMemory()
        if actorShared and baseline is not None: baseline.shareMemory()

        self.workers = workers = dict()
        s1, s2 = config["agent"].get("seedTrain", TRAIN_SEED), config["agent"].get("seedTest", TEST_SEED)
        for i in range(nWorkers):
            nconfig = config.copy()
            nconfig["agent"]["id"] = i
            nconfig["env"]["seedTrain"] = s1 + i + 1
            nconfig["env"]["seedTest"] = s2 + i + 1
            nconfig['policy']['policyClass'] = nconfig['policy']['workerPolicyClass']
            if policy is not None:
                if not actorShared:
                    nActor = cloneNet(actor)
                    nBl = cloneNet(baseline) if baseline is not None else None
                # serializes the current actor and baseline to the agent init on thread
                # within the workerPolicy serialization
                workerPolicy = createPolicy(nconfig, nActor, baseline = nBl)
            worker = Worker(ragnt.remote(nconfig, workerPolicy, envMaker), i)
            workers[i] = worker

        self.set4Ray()
    
    @property
    def device(self):
        if self.policy is not None:
            return self.policy.device
        else:
            return DEVICE_DEFT

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
            w.ref = w().close.remote()
        self.syncResolve()
        for w in self.workers.values():
            del w.worker
        ray.shutdown()

    def set4Ray(self):
        wrks = []
        for w in self.workers.values():
            w.ref = w().set4Ray.remote()
            wrks.append(w)
        self.syncResolve(wrks)

    def __repr__(self) -> str:
        s = self.name + ', with %d workers' % len(self.workers)
        return s

class agentSync(agentMaster):
    name = 'Agent master sync'

    def reset(self):
        for worker in self.workers.values():
            worker.ref = worker().reset.remote()
        results = self.syncResolve()
        self.lastObs = mergeDicts(*results, targetDevice = self.device)
        return results

    def fullStep(self):
        wrks = []
        for w in self.workers.values():
            w.ref = w().fullStep.remote()
            wrks.append(w)
        
        return mergeDicts(*self.syncResolve(wrks), targetDevice = self.device)

    def getEpisode(self, random: bool = False, device = None):
        wrks = []
        for w in self.workers.values():
            w.ref = w().getEpisode.remote()
            wrks.append(w)

        return self.syncResolve(wrks)
        device = device if device is not None else self.device
        return mergeDicts(*self.syncResolve(), targetDevice = device)

    def getBatch(self, size: int, proportion: float = 1.0):
        pass

    def sync(self, batch):
        self.policy.update(batch)

        state = self.policy.currentState()
        for w in self.workers.values():
            w.ref = w().loadState.remote(state)
        
        self.syncResolve()

    def getParams(self):
        wrks = []
        for w in self.workers.values():
            w.ref = w().getParams.remote()
            wrks.append(w)

        return self.syncResolve(wrks)

    def updateParams(self, piParams, blParams):
        piParams_ = ray.put(piParams)
        blParams_ = ray.put(blParams)
        wrks = []
        for w in self.workers.values():
            w.ref = w().updateParams.remote(piParams_, blParams_)
            wrks.append(w)

        self.syncResolve(wrks)

    def test(self, iters: int = TEST_N_DEFT, progBar: bool = False):
        '''
            From Agent base test method
        '''
        itersPerWorker = ceil(iters / self._nWorkers)
        wrks = []
        for w in self.workers.values():
            w.ref = w().test.remote(itersPerWorker)
            wrks.append(w)

        results = mergeResults(*self.syncResolve(wrks))

        if self.tbw != None:
            self.tbw.add_scalar("test/mean Return", results['mean_return'], self.testCalls)
            self.tbw.add_scalar("test/mean Steps", results['mean_steps'], self.testCalls)
            self.tbw.add_scalar("test/std Return", results['std_return'], self.testCalls)
            self.tbw.add_scalar("test/std Steps", results['std_steps'], self.testCalls)
            self.tbw.add_scalar("test/max Return", results['max_return'], self.testCalls)
            self.tbw.add_scalar("test/min Return", results['min_return'], self.testCalls)
            self.tbw.add_scalar("test/tests Achieved", results['tot_tests'], self.testCalls)
        self.testCalls += 1

        return results
