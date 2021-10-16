from torch._C import Value
from rofl.config.config import createPolicy, createAgent
from rofl.functions.coach import singlePathRolloutMulti
from rofl.functions.const import *
from rofl.functions.functions import ceil, deepcopy
from rofl.functions.dicts import composeObs, mergeDicts, mergeResults
from rofl.policies.base import Policy
from rofl.utils.memory import multiMemory
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
    """
        Main class to create and manage a pool of ray workers.

        Notes:
        - workers will be seeded with the base of the main seed, but results
            will vary with the number of workers. To reproduce both need to be
            equal.
        - Needs in agent config a wokerClass, as the main agent class is this (agentMaster).
        - Needs in policy config a workerPolicyClass, same or not. A new subclass from a policy
            could be used to save resources, eg. omits to create an optimizer or else.
    """
    name = 'Agent Master v1'
    isMulti = True

    def __init__(self, config: dict, policy: Policy, envMaker, **kwargs):

        self.policy = policy
        self.config = config

        self.tbw = kwargs.get('tbw')
        self.testCalls = 0

        nWorkers = config["agent"].get("workers", NCPUS)
        nWorkers += 1 if nWorkers == 1 else 0
        self._nWorkers = nWorkers = NCPUS if nWorkers > NCPUS or nWorkers < 1 else nWorkers

        import rofl.agents as agents
        agentClass = getattr(agents, config['agent']['workerClass']) # should raise error when ill config
        ray.init(num_cpus = nWorkers)
        ragnt = ray.remote(agentClass)

        workerPolicy = None
        nActor = policy.actor
        nBl = getattr(policy, 'baseline') if not getattr(policy, 'actorHasCritic', True) else None

        self.workersDict = workersD = dict()
        self.workers = workersL = []
        s1, s2 = config["agent"].get("seedTrain", TRAIN_SEED), config["agent"].get("seedTest", TEST_SEED)
        for i in range(nWorkers):
            nconfig = deepcopy(config) # TODO, deep copy
            nconfig["agent"]["id"] = i
            nconfig["env"]["seedTrain"] = s1 + i + 1
            nconfig["env"]["seedTest"] = s2 + i + 1
            nconfig['policy']['policyClass'] = nconfig['policy']['workerPolicyClass']
            workerPolicy = createPolicy(nconfig, nActor, baseline = nBl)
            worker = Worker(ragnt.remote(nconfig, workerPolicy, envMaker), i)
            workersD[i] = worker
            workersL.append(worker)
    
    @property
    def device(self):
        pi = self.policy
        if pi is not None:
            return pi.device
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
        for w in self.workers:
            status = w.status
            if status == READY:
                ready.append(w)
            elif status == WORKING:
                # try to resolve
                if timeout > 0:
                    if w.resolve(timeout):
                        done.append(w)
                working.append(w)
            elif status == DONE:
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
            working = [w for w in self.workers if w.status == WORKING]
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
        for w in self.workers:
            w.ref = w().close.remote()
        self.syncResolve()
        for w in self.workers:
            del w.worker
        ray.shutdown()

    def __repr__(self) -> str:
        s = self.name + ', with %d workers' % len(self.workers)
        return s

class agentSync(agentMaster):
    name = 'Agent master sync'

    def reset(self):
        wrks = []
        for worker in self.workers:
            worker.ref = worker().reset.remote()
            wrks.append(worker)
        return self.syncResolve(wrks)

    def envStep(self, actions, **kwargs):
        wrks = []
        actions = ray.put(actions)
        for w in self.workers:
            w.ref = w().envStep.remote(actions, **kwargs)
            wrks.append(w)
        return self.syncResolve(wrks)

    def fullStep(self, random: bool = False, **kwargs):
        wrks = []
        for w in self.workers:
            w.ref = w().fullStep.remote(random, **kwargs)
            wrks.append(w)
        
        return self.syncResolve(wrks)

    def getEpisode(self, random: bool = False, device = None) -> list[dict]:
        wrks = []
        #device_ = ray.put(device if device is not None else self.device) # cpu only still 
        for w in self.workers:
            w.ref = w().getEpisode.remote()
            wrks.append(w)

        return self.syncResolve(wrks)

    def getGrads(self, random: bool = False):
        wrks = []
        for w in self.workers:
            w.ref = w().calculateGrad.remote()
            wrks.append(w)

        return self.syncResolve(wrks)

    def getBatch(self, size: int, proportion: float = 1.0):
        pass

    def sync(self, batch):
        self.policy.update(batch)
        state = self.policy.currentState()
        for w in self.workers:
            w.ref = w().loadState.remote(state)
        
        self.syncResolve()

    def getParams(self):
        wrks = []
        for w in self.workers:
            w.ref = w().getParams.remote()
            wrks.append(w)

        return self.syncResolve(wrks)

    def updateParams(self, piParams, blParams):
        piParams_ = ray.put(piParams)
        blParams_ = ray.put(blParams) if blParams != [] else []
        wrks = []
        for w in self.workers:
            if blParams == []: # ray.put has a cost per call, even for []
                w.ref = w().updateParams.remote(piParams_)
            else:
                w.ref = w().updateParams.remote(piParams_, blParams)
            wrks.append(w)

        self.syncResolve(wrks)

    def test(self, iters: int = TEST_N_DEFT, progBar: bool = False):
        '''
            From Agent base test method
        '''
        itersPerWorker = ceil(iters / self._nWorkers)
        wrks = []
        for w in self.workers:
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

class agentMultiEnv(agentSync):
    name = 'Agent multi env'

    def __init__(self, config, policy, envMaker, **kwargs):
        super().__init__(config, policy, envMaker, **kwargs)
        # More like BaseAgent, but to manage multiAgent
        keys = [('action', I_TDTYPE_DEFT)] if self.policy.discrete else [('action', F_TDTYPE_DEFT)]
        self.memory = multiMemory(config, *keys)
        
        nConfig = config.copy()
        nConfig['agent']['agentClass'] = config['agent']['workerClass']
        self.leadAgent = createAgent(nConfig, policy, envMaker, **kwargs)
        self.nSteps = config['agent']['nstep']
        if self.nSteps < 0:
            raise ValueError('For this agent, nstep needs to be positive to impose a nstep return')

        self.stepReady = False
        self.lastObs, self.lastDones = None, None
        self.lastIds = None
        
        self.needsLogProb = needProbs = config['agent']['need_log_prob']
        if needProbs and not policy.stochastic:
            raise AttributeError('%s cannot provide log_probs as requested'%policy)
        self.needsObsValue = needValue = config['agent']['need_obs_value']
        if needValue and not policy.valueBased:
            raise AttributeError('%s cannot provide value as requested'%policy)

    def test(self, iters: int = TEST_N_DEFT, progBar: bool = False):
        return self.leadAgent.test(iters=iters, progBar=progBar)

    def rndAction(self):
        """
            Returns a random action from the gym space
        """
        actions = []
        rndFun = self.leadAgent.rndAction
        for _ in self.workers:
            actions.append(rndFun())
        return actions

    def getEpisode(self, random: bool = False, device = None) -> list[dict]:
        pi, memory = self.policy, self.memory
        device = pi.device if device is None else device
        singlePathRolloutMulti(self, self.nSteps, random)
        return memory.getEpisodes(device, pi.keysForUpdate)

    def reset(self):
        observations = super().reset()
        self.lastObs, self.lastDones, self.lastIds = composeObs(*observations, device = self.device)
        self.memory.reset()
        self.stepReady = True

        return observations

    def fullStep(self, random: bool = False):
        if not self.stepReady:
            self.reset()

        needProbs, needValues = self.needsLogProb, self.needsObsValue
        pi, observation, ids = self.policy, self.lastObs, self.lastIds

        if random:
            actions = self.rndAction()
        elif needProbs and not needValues:
            actions, logProbs = pi.getActionWProb(observation)
        elif not needProbs and needValues:
            actions, values = pi.getActionWVal(observation)
        elif needProbs and needValues:
            actions, values, logProbs = pi.getAVP(observation)
        else:
            actions = pi.getAction(observation)

        observations = self.envStep(actions, ids = ids)

        if needValues or needProbs:
            iDS = {}
            for dict_ in observations:
                iDS[dict_['id']] = dict_
        if needValues:
            for value, iD in zip(values, ids):
                dict_ = iDS[iD]
                dict_['obs_value'] = value
        if needProbs:
            for prob, iD in zip(logProbs, ids):
                dict_ = iDS[iD]
                dict_['log_prob'] = prob

        self.lastObs, self.lastDones, self.lastIds = composeObs(*observations, device = self.device) # compose to have a state to process actions with
        return observations
        self.memory.add(*observations) # the rest of the info goes to the memories in raw

        