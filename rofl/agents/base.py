from warnings import WarningMessage
from gym.core import RewardWrapper
from rofl.functions.const import *
from rofl.functions.torch import *
from rofl.functions.dicts import obsDict
from rofl.functions.gym import doWarmup, noOpSample
from rofl.policies.dummy import dummyPolicy
from gym import Env
from abc import ABC

class Agent(ABC):
    """
    Base class for actors. 

    Must be initiated with:
    - A Configuration dict
    - A Environment Maker function
    - Policy (to enable fullStep method)
    Optionals:
    - tensorboard writer in .tbw
    
    Main Methods:
    - getBatch: With a size and proportion, returns a
        trajectory dict
    - envStep: Recieves an action and executes into the selected
        environment
    - fullStep: Process an action from the policy and calls 
        envStep
    - test: executes a test, returns results
    - reset: resets the state to start over

    Customized methods:
    - processObs: Process the actual observation
        returns the new observation or the same. If required
    - processReward: Process the reward from the
        envStep operation. If required
    - isTerminal: Outputs if the actual state is a terminal
        type or not. If required
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

    Properties:
    - observation
    - status

    Other methods: (in progress)
    - currentState: returns a dict
    - loadState: Loads a state dict
    """
    name = "BaseAgent"
    environment = None
    policy = None
    config = None
    env, envTest = None, None
    twb, testCalls = None, 0
    _ac, _nstep, done = 0.0, 0, True
    lastObs, lastReward, lastInfo =  None, 0.0, {}
    _totSteps, _totEpisodes = 0, -1

    def __init__(self):
        if self.name == "BaseAgent":
            raise NameError("New agent should be called different to BaseAgent")
        if self.config is None or not isinstance(self.config, dict):
            raise ValueError("Agent needs .config as a dict")
        if not isinstance(self.env, Env):
            raise ValueError("At least self.env should be defined with a gym's environment")
        if self.policy is None:
            self.policy = dummyPolicy(self.env)
            print("Warning, agent working with a dummy plug policy!")
        if self.envTest is None:
            self.envTest = self.env
            print("Warning, environment test not given. Test will be evaluated with the train environment!")
        # some administrative
        self.workerID = self.config["agent"].get("id", 0)
        self._noop_ = noOpSample(self.env)
        
    def currentState(self):
        """
            Returns a dict with all the required information
            of its state to start over or just to save it.
        """
        return dict()

    def loadState(self, newState):
        """
            Form a dictionary state, loads all the values into
            the agent.
            Must verify the name of the agent is the same and the
            type.
        """
        return NotImplementedError

    def test(self, iters:int = TEST_N_DEFT, prnt:bool = False):
        """
            Main method to evaluate any policy for the envinroment, 
            this should not be changed.

            parameters
            ----------
            iters: int
                Number of test to execute. More the better
            prnt: bool
                If it should print the main results or not
            
            returns
            -------
            meanAccReward: float
                Mean of the accumulated reward
            stdMean: float
                Variance corresponding to the mean from all the
                tests done by iters
            meanSteps: float
                Mean number of steps by the policy in the environment
            stdMeanSteps: float
                Variance corresponding to the mean of steps from all the
                tests done
        """
        # Init
        env = self.env if self.envTest is None else self.envTest
        accRew, steps, testReg = np.zeros((iters)), np.zeros((iters)), 0
        maxReturn, minReturn = -np.inf, np.inf
        self.prepareCustomMetric()
        episodeLen = self.config["env"].get("max_length", MAX_EPISODE_LENGTH)
        testLen, totSteps, stepsDone = self.config["train"].get("max_steps_test", -1), 0, False
        proc = self.processObs
        # Set policy to test mode
        self.prepareTest()
        self.policy.test = True
        # Iterations for the n_test
        for test in range(iters):
            testDone, testGain, testSteps = False, 0.0, 0
            obs = env.reset()
            obs = proc(obs, reset = True)
            while not testDone:
                action = self.policy.getAction(obs)
                nextObs, reward, done, _ = env.step(action)
                self.calculateCustomMetric(env, reward, done)
                testGain += reward
                testSteps += 1
                totSteps += 1
                # Terminal condition for episode
                testDone = done
                if episodeLen > 0 and testSteps >= episodeLen:
                    testDone = True
                ## Condition by max steps per test if valid. 
                ## Can happen only if there is at least one result score
                if testLen > 0 and totSteps >= testLen and totSteps != testSteps:
                    stepsDone = True
                    break
                obs = proc(nextObs)
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
        # Printing
        if prnt:
            print("Test results mean Return:{%.2f}, mean Steps:{%.2f}, std Return:{%.3f}, std Steps:{%.3f}" % (\
                meanAccReward, meanSteps, stdMean, stdMeanSteps))
        # Generating results
        results = {"mean_return": meanAccReward,
                "std_return": stdMean,
                "mean_steps": meanSteps, 
                "std_steps": stdMeanSteps,
                "custom": self.reportCustomMetric(),
                "max_return": maxReturn,
                "min_return": minReturn,
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

    def processObs(self, obs, reset:bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here

            parameters
            ----------
            obs
                The observations as they come from the environment. 
                Usually ndarray's.

            reset: bool
                If needed a flag that this observation is from a new
                trajectory. As to reset the pipeline or not effect.
        """
        return obs

    def processReward(self, reward, **kwargs):
        return reward

    def isTerminal(self, obs, done, info, **kwargs):
        return done
    
    def envStep(self, action, **kwargs):
        """
            Develops an action in the environment with
            the action. Adds the observation and reward to the agent's
            administrative state.

            return:
            infoDict
        """
        if self.done:
            return self.reset()

        env = self.env
        procObs, procRew, procTer = self.processObs, self.processReward, self.isTerminal
        self._totSteps += 1

        self._nstep += 1
        # TODO create a function that process actions for the policy
        # Process action from a batch
        ids = kwargs.get("ids")
        if ids is not None:
            # TODO this is not a great way
            for i, d in enumerate(ids):
                if d == self.workerID:
                    action = action[i]
                    break

        obs, reward, done, info = env.step(action)

        pastObs = self.lastObs
        self.lastObs = obs = procObs(obs)
        self.lastReward = reward = procRew(reward)
        self.lastInfo = info
        self.done = done = procTer(obs, done, info)

        self._ac += reward
        self._totEpisodes += 1 if done else 0

        return obsDict(pastObs, action, reward, 
                        self._nstep, done, info, 
                        accumulate_reward = self._ac,
                        id = self.workerID) 

    def fullStep(self, random = False, **kwargs):
        """
            Does a full step on the environment calling the envStep
            method with an action from the policy if available or a
            noOp action.
        """
        action = self.policy.getAction(self.lastObs) if not random else self.exploratoryAction(self.lastObs)
        return self.envStep(action)

    def reset(self):
        """
            Resets the main environment and emits an observation
            withour any warmup.

            return:
            infoDict
        """
        self._nstep, self._ac = 0, 0.0
        return self._startEnv(warmup = self.config["env"].get("warmup"))

    def _startEnv(self, warmup = None):
        env = self.env
        self.done = False

        if warmup is not None:
            obs, steps, action = doWarmup(warmup, env, self.config["env"])
        else:
            obs, action = env.reset(), self._noop_
        self.lastObs = obs = self.processObs(obs, True)

        return obsDict(obs, action, 0.0, 0, False, 
                        accumulate_reward = self._ac,
                        id = self.workerID)

    def exploratoryAction(self, observation):
        """
            Defines an exploration strategy for the agent.

            Default
            Return a random action from the gym space
        """
        return self.env.action_space.sample()

    def getBatch(self, size: int, proportion: float = 1.0):
        """
            Prepares and return a batch of information or trajectories
            to feed the algorithm to update the policy.
        """
        return dict()

    def __repr__(self):
        s = "Agent {}\nFor environment {}\nPolicy {}".format(self.name, 
            self.environment, self.policy)
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
        