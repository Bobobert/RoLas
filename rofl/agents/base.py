from typing import Union
from gym import Env
from tqdm import tqdm

from rofl.config.types import AgentType, PolicyType
from rofl.functions.const import TEST_N_DEFT, DEVICE_DEFT, TENSOR
from rofl.functions.functions import assertProb, newZero, np, noGrad, ceil, deepcopy
from rofl.functions.torch import tryCopy
from rofl.functions.dicts import ObsDict
from rofl.functions.gym import doWarmup, noOpSample
from rofl.functions.coach import singlePathRollout

class BaseAgent(AgentType):
    """
    Base class for agents. 

    Parameters
    ----------
    - config: dict
        A configuration dictionary
    - policy: Policy
        A Policy Type object (NoneType results in No-operation policy)
    - envMaker: function
        Environment maker function from the rofl.envs
    - tbw (optional): tensorboard writer
    
    Main Methods
    ------------
    - getBatch: With a size and proportion, returns an obsDict
        as batch
    - getEpisode: Returns a full episode with the calculated 
        returns in a obsDict
    - envStep: Recieves an action and executes into the selected
        environment
    - fullStep: Process an action from the policy and calls 
        envStep
    - test: executes a test, returns results
    - reset: resets the state to start over
    - rndAction: returns a sample from the environment's action space

    Customizable methods
    ------------------
    - initAgent: Initialize custom variables
    - processObs: Process the actual observation
        returns the new observation or the same. If required
    - processReward: Process the reward from the
        envStep operation. If required
    - isTerminal: Outputs if the actual state is a terminal
        type or not. If required
    - For testing:
        - prepareTest: if necesary prepares the agent to execute
            a test, if not does nothing
        - prepareAfterTest: Continuation to prepareTest, if necesary, 
            set agent state after a test
        - prepareCustomMetric: if needed, the agent can have a custom
            method to calculate a metric while testing only. In training
            can be writen inside the policy. This method will be called 
            each time .test is called 
        - calculateCustomMetric: if needed the previous one, each step of 
            the test.
        - reportCustomMetric: return the custom metric    


    Other methods: (in progress)
    -------------
    - currentState: returns a dict
    - loadState: Loads a state dict
    """
    name = "BaseAgent"

    def __init__(self, config: dict, policy: PolicyType, envMaker, **kwargs):

        self.testCalls = 0
        self._acReward, self._agentStep, self._envStep = 0.0, 0, 0
        self.done, self._reseted = True, False
        self.lastObs, self.lastReward, self.lastAction, self.lastInfo =  None, 0.0, None, {}
        self.lastRawObs, self.lastDone = None, None
        self.zeroObs = None
        self._totSteps, self._totEpisodes, self.memory = 0, 0, None

        if self.name == "BaseAgent":
            raise NameError("New agent should be called different to BaseAgent")

        if config is None or not isinstance(config, dict):
            raise ValueError("Agent needs config as a dict")

        self.config = config
        self.tbw = kwargs.get('tbw')
        self.env, self.trainSeed = envMaker(config["env"]["seedTrain"])
        self.envTest, self.testSeed = envMaker(config["env"]["seedTest"])
        self.noOp = noOpSample(self.env)
        self.actionSpace = self.env.action_space

        try:
            self.envName = self.env.name
        except AttributeError:
            self.envName = config['env']['name']

        if policy is None:
            from rofl.policies.base import DummyPolicy
            self.policy = policy = DummyPolicy(self.noOp)
            self.needsLogProb = self.needsObsValue = needProbs = needValue = False
            print("Warning, agent working with a dummy plug policy!")
        else:
            self.policy = policy
            self.needsLogProb = needProbs = config['agent']['need_log_prob']
            self.needsObsValue = needValue = config['agent']['need_obs_value']
        policy.rndFunc = self.rndAction
        self.keysForBatches = policy.keysForUpdate
        
        # some administrative
        self.workerID = config["agent"].get("id", 0)
        self.gamma = assertProb(config["agent"]["gamma"])
        self.lmbd = assertProb(config["agent"].get("lambda", 1.0))
        self.gae = config["agent"].get("gae", False)
        self.maxEpLen = config["env"].get("max_length", -1)
        self.warmup = config["env"].get("warmup")

        if needProbs and not policy.stochastic:
            raise AttributeError('%s cannot provide log_probs as requested'%policy)
        if needValue and not policy.valueBased:
            raise AttributeError('%s cannot provide value as requested'%policy)
        
        self.initAgent(**kwargs)

    def initAgent(self, **kwargs):
        """
            If needed, write additional initialization for 
            parameters and functions setup. Usually this is done just once
            per agent, agents should live as long as the experiment, so the
            cost should not matter.

            returns
            -------
            None
            
        """
        pass

    def currentState(self):
        """
            Returns a dict with all the required information
            of its state to start over or just to save it.
        """
        return {"name": self.name, "envName": self.envName,
                "config": self.config, "policy_state": self.policy.currentState(),
                "env_obs": self.lastObs, "env_state":None, "env_done": self.done,
                "env_steps": self._envStep, "agent_step": self._agentStep,
                "accumulated_reward": self._acReward, "reward": self.lastReward,
                "env_info":self.lastInfo}

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the agent.
            Must verify the name of the agent is the same and the
            type.
        """
        if newState["name"] != self.name:
            raise ValueError("Agent type must be the same!")
        if newState["envName"] != self.envName:
            raise ValueError("Environment should be the same")
        # Not a copy as many contain a VariableType and else
        self.config = newState["config"] 
        self.policy.loadState(newState["policy_state"])
        # Agent variables
        self.lastObs = tryCopy(newState["env_obs"])
        self.lastReward = newState["reward"]
        self.lastInfo = deepcopy(newState["env_info"])
        self.done = newState["env_done"]
        self._acReward = newState["accumulated_reward"]
        self._envStep, self._agentStep = newState["env_steps"], newState["agent_step"]

    def test(self, iters:int = TEST_N_DEFT, prnt:bool = False, progBar: bool = False):
        """
            Main method to evaluate any policy for the envinroment, 
            this should not be changed.

            parameters
            ----------
            - iters: int
                Number of test to execute. More the better
            - prnt: bool
                If it should print the main results or not
            - progBar:
                Default False. To print a tqdm bar progress of the
                episodes per test.
            
            returns
            -------
            - meanAccReward: float
                Mean of the accumulated reward
            - stdMean: float
                Variance corresponding to the mean from all the
                tests done by iters
            - meanSteps: float
                Mean number of steps by the policy in the environment
            - stdMeanSteps: float
                Variance corresponding to the mean of steps from all the
                tests done
        """
        # Init
        assert iters > 0, "iters is expected to be a positive number"
        env = self.env if self.envTest is None else self.envTest
        accRew, steps, testReg = np.zeros((iters)), np.zeros((iters)), 0
        maxReturn, minReturn = -np.inf, np.inf
        self.prepareCustomMetric()
        episodeLen = self.maxEpLen
        testLen, totSteps, stepsDone = self.config["train"]['max_steps_per_test'], 0, False
        proc = self.processObs
        # Set policy to test mode
        self.prepareTest()
        self.policy.test = True
        # Iterations for the n_test
        iters = range(iters)
        if progBar:
            iters = tqdm(iters, desc='Testing agent', unit='episode')
        for test in iters:
            testDone, testGain, testSteps = False, 0.0, 0
            obs = env.reset()
            obs = proc(obs, {}, False, reset = True)
            while not testDone:
                action = self.policy.getAction(obs)
                nextObs, reward, done, info = env.step(action)
                self.calculateCustomMetric(env, reward, done)
                testGain += reward
                testSteps += 1
                totSteps += 1
                # Terminal condition for episode
                testDone = done
                if episodeLen > 0 and testSteps >= episodeLen:
                    testDone = True
                ## Condition by max steps per test if valid. 
                ## Can happen only if there is at least one result score, changed
                if testLen > 0 and totSteps >= testLen:
                    if totSteps == testSteps:
                        print('Testing: Failed to complete an episode!\
                            Budget steps(%d) spent completely' % testLen)
                        testReg = 1
                    stepsDone = True
                    break
                obs = proc(nextObs, info, done)
            # Continuation on max steps test for the main loop
            if stepsDone and (testDone != True):
                break
            # Processing metrics
            accRew[test] = testGain
            steps[test] = testSteps
            if testGain > maxReturn:
                maxReturn = testGain
            if testGain < minReturn:
                minReturn = testGain
            testReg += 1
        # calculate means and std
        meanAccReward, meanSteps = np.mean(accRew[:testReg]), np.mean(steps[:testReg])
        stdMean, stdMeanSteps = np.std(accRew[:testReg]), np.std(steps[:testReg])
        # Register to tb writer if available
        if self.tbw != None:
            self.tbw.add_scalar("test/mean Return", meanAccReward, self.testCalls)
            self.tbw.add_scalar("test/mean Steps", meanSteps, self.testCalls)
            self.tbw.add_scalar("test/std Return", stdMean, self.testCalls)
            self.tbw.add_scalar("test/std Steps", stdMeanSteps, self.testCalls)
            self.tbw.add_scalar("test/max Return", maxReturn, self.testCalls)
            self.tbw.add_scalar("test/min Return", minReturn, self.testCalls)
            self.tbw.add_scalar("test/tests Achieved", testReg, self.testCalls)
        # Returning state
        self.prepareAfterTest()
        self.policy.train = True
        # Printing
        if prnt:
            print("Test results mean Return: %.2f, mean Steps: %.2f, std Return: %.3f, std Steps: %.3f" % (\
                meanAccReward, meanSteps, stdMean, stdMeanSteps))
        # Generating results
        results = {
            "mean_return": meanAccReward,
            "std_return": stdMean,
            "mean_steps": meanSteps, 
            "std_steps": stdMeanSteps,
            "custom": self.reportCustomMetric(),
            "max_return": maxReturn,
            "min_return": minReturn,
            'tot_tests': testReg,
            }
        self.testCalls += 1
        return results

    def prepareTest(self):
        """
            If the agent needs to prepare to save or load 
            anything before a test. Write it here
        """
        pass
    
    def prepareAfterTest(self):
        """
            If the agent needs to prepare to save or load 
            anything after a test. Write it here
        """
        pass

    def processObs(self, obs, info: dict, done: bool, reset:bool = False) -> TENSOR:
        """
            If the agent needs to process the observation of the
            environment. Write it here

            parameters
            ----------
            - obs
                The observations as they come from the environment. 
                Usually ndarray's.
            -info
                Dict from the environment step.
            - done: bool
                From isTerminal method
            - reset: bool
                If needed a flag that this observation is from a new
                trajectory. As to reset the pipeline or not effect.
            
            returns
            -------
            - torch.Tensor
        """
        print("Warning: Agent %s. processObs method not defined!" % self.name)
        return obs

    def processReward(self, obs, reward: float, info: dict, done: bool) -> Union[int, float]:
        """
            If the agent needs to process the reward of the
            environment. Write it here

            parameters
            ----------
            - obs:
                The unprocessed observation from the environment step
                method
            - reward: int, float
            - info: dict
            - done: bool
                From isTerminal method

            returns
            -------
            - float
        """
        return reward

    def isTerminal(self, obs, reward: float, info: dict, done: bool) -> bool:
        """
            If the agent has special conditions to mark
            a state as terminal. Write it here.

            Default(super) behavior, if config-> env ->max_length 
            is greater than 0 and self._envStep is equal or greater
            then is marked as terminal. 
            
            If the warmup adds steps to the environment those are consider
            as well. Can use self._agentStep to watch the agent's steps
            in the actual episode on the environment.

            parameters
            ----------
            - obs:
                The unprocessed observation from the environment step
                method
            - reward: int, float
            - info: dict
            - done: bool

            returns
            -------
            - bool
        """
        self.done = done
        if self.maxEpLen > 0 and self._envStep >= self.maxEpLen:
            self.done = True
        return self.done
    
    def envStep(self, action, **kwargs):
        """
            Develops an action in the environment with
            the action. Adds the observation and reward to the agent's
            administrative state.

            parameters
            ----------
            action: int or ndarray

            returns
            --------
            obsDict
        """
        if self.done:
            return self.reset()
        if self._reseted:
            self._reseted = False

        # Process action from a batch
        for i, d in enumerate(kwargs.get("ids", [])):
            if d == self.workerID:
                action = action[i]
                break

        obs, reward, done, info = self.env.step(action)

        # Process the outputs
        pastObs = self.lastObs
        self.lastInfo, self.lastAction = info, action
        self.lastRawObs = obs
        self.lastDone = done = self.isTerminal(obs, reward, info, done)
        self.lastObs = nextObs = self.processObs(obs, info, done)
        self.lastReward = reward = self.processReward(obs, reward, info, done)

        # Incrementals
        self._totSteps += 1
        self._agentStep += 1
        self._envStep += 1
        self._acReward += reward
        self._totEpisodes += 1 if done else 0

        return ObsDict(pastObs, nextObs, action, reward, 
                        self._agentStep, done, info, 
                        accumulate_reward=self._acReward,
                        id=self.workerID) 

    def fullStep(self, random:bool = False, **kwargs):
        """
            Does a full step on the environment calling the envStep
            method with an action from the policy, or random action.

            parameters
            ----------
            random: bool
                If true overrides the policy action process and uses
                a sample from the environment's action space

            returns
            -------
            obsDict
        """
        needProbs, needValues = self.needsLogProb, self.needsObsValue
        pi, observation = self.policy, self.lastObs

        if self.done:
            action = None
        elif random:
            action = self.rndAction()
        elif needProbs and not needValues:
            action, logProb = pi.getActionWProb(observation)
        elif not needProbs and needValues:
            action, value = pi.getActionWVal(observation)
        elif needProbs and needValues:
            action, value, logProb = pi.getAVP(observation)
        else:
            action = pi.getAction(observation)
        
        infoDict = self.envStep(action)

        if needValues:
            if self._reseted:
                # this should happen just after a reset call
                value = pi.getValue(infoDict['next_observation'], infoDict['action'])
            infoDict['obs_value'] = value

        if needProbs:
            if self._reseted:
                # after a reset the actor was never samppled, then the noOP or whatever
                # 'joker' action from reset function needs to be sampled in this version
                # to keep the calculations accurate
                with noGrad():
                    logProb = pi.getProb4Action(infoDict['next_observation'], infoDict['action'])
            infoDict['log_prob'] = logProb

        return infoDict

    def reset(self):
        """
            Resets the main environment and emits an observation
            withour any warmup.

            returns
            --------
            obsDict
        """
        return self._startEnv(warmup = self.warmup)

    def _startEnv(self, warmup = None):
        env = self.env
        
        if warmup is not None:
            obs, self._envStep, action = doWarmup(warmup, env, self.config["env"])
        else:
            obs, self._envStep, action = env.reset(), 0, self.noOp

        self.lastObs = obs = self.processObs(obs, {}, False, True)
        zeroObs = self.zeroObs
        if zeroObs is None:
            self.zeroObs = zeroObs = newZero(obs)
        self.lastReward, self.lastInfo, self.lastAction = 0.0, {}, action
        self._agentStep, self._acReward = 0, 0.0
        self._reseted, self.done = True, False

        return ObsDict(zeroObs, obs, action, 0.0, 0, False, 
                        accumulate_reward=self._acReward,
                        id=self.workerID)
    
    def rndAction(self, *args):
        """
            Returns a random action from the gym space
        """
        return self.actionSpace.sample()

    def getBatch(self, size: int, proportion: float = 1.0, random = False,
                        device = DEVICE_DEFT, progBar: bool = False):
        """
            Prepares and return a batch of information or trajectories
            to feed the algorithm to update the policy.

            parameters
            ----------
            - size: int
                Size of the batch to return
            - proportion: float
                The ratio sample size / steps required.
                proportion = 0.5 means the agent
                needs to do double the amount of steps respect to
                the batch size.
            - random: bool
                Default False. If the action are generated as samples
                of the environments action space.
            - device: torch.device
            - progBar: bool
                Default False. To show a progress bar using tdqm

            Returns
            -------
            batch obsDict
        """
        assert proportion > 0, "Proportion needs to be positive"
        
        memory = self.memory
        if memory is None:
            if proportion > 1.0:
                raise ValueError("Agent doesnt have a previous memory, the sample wont happen.")
            from rofl.utils.memory import SimpleMemory
            memory = SimpleMemory(self.config).reset()

        iter = range(ceil(size / proportion))
        if progBar:
            iter = tqdm(iter, desc='Generating batch', unit='envStep')
        for _ in iter:
            memory.add(self.fullStep(random=random))
        return memory.sample(size, device, self.keysForBatches)

    def getEpisode(self, random:bool = False, device = None):
        """
            Develops or complete an entire episode in the environment.
            Calculates the returns of the steps with a episodicMemory.

            returns
            -------
            obsDict
        """
        memory = singlePathRollout(self, random=random, reset=True)
        device = device if device is not None else self.device
        return memory.getEpisode(device, self.keysForBatches)

    def __repr__(self):
        s = "Agent {}\nFor environment {}\n{}".format(self.name, 
            self.envName, self.policy)
        return s

    def prepareCustomMetric(self):
        """
            This method will be called once each time .test is called.
            It should prepare the variables in the object to report the
            results after each test.
            calculateCustomMetric is the one called each time a test has 
            reached a terminal state.
        """
        pass

    def calculateCustomMetric(self, env:Env, reward: float, done: bool):
        """
            From the state in test, and the self.envTest. Here custom code
            can be writen to calculate metrics differente from the test.
        """
        pass

    def reportCustomMetric(self):
        """
            From the custom metric calculation, one can pass anything that can
            be saved to the test method. This will be stored on the test results
            dict.
        """
        return {}

    @property
    def device(self):
        if self.policy is None:
            return None
        return self.policy.device

    def close(self):
        """
            Meant to close the agent's environments.
        """
        self.env.close()
        if self.envTest != None:
            self.envTest.close()
        