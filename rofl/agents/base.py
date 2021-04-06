from rofl.functions.const import *
from rofl.functions.torch import *
from gym import Env
from abc import ABC

class Agent(ABC):
    """
    Base class for actors. 

    Must be initiated with:
    - A Configuration dict
    - A Environment Maker function
    - A policy
    Optionals:
    - tensorboard writer in .tbw
    
    Main Methods:
    - getBatch: With a size and proportion, returns a
        trajectory dict
    - processObs: if required process the actual observation
        returns the new observation or the same
    - test: executes a test, returns results
    - prepareTest: if necesary prepares the agent to execute
        a test, if not does nothing
    - prepareAfterTest: Continuation to prepareTest, if necesary, 
        set agent state after a test
    - reset: resets the state to start over

    Other methods: (in progress)
    - prepareCustomMetric: if needed, the agent can have a custom
        method to calculate a metric while testing only. In training
        can be writen inside the policy. This method will be called 
        each time .test is called 
    - calculateCustomMetric: if needed the previous one, this is called
        at the end of each test for the terminal state.
    - reportCustomMetric: return the custom metric
    - currentState: returns a dict
    - loadState: Loads a state dict
    """
    name = "BaseAgent"
    environment = None
    policy, config = None, None
    env, envTest = None, None
    twb = None
    def __init__(self):
        if self.name == "BaseAgent":
            raise NameError("New agent should be called different to BaseAgent")
        if self.config is None or not isinstance(self.config, dict):
            raise ValueError("Agent needs .config as a dict")
        if not isinstance(self.env, Env):
            raise ValueError("At least self.env should be defined with a gym's environment")
        if self.policy is None:
            raise ValueError("A policy must be given")
        if self.envTest is None:
            print("Warning, environment test not given. Test will be evaluated with the train environment!")

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
        accRew, steps = np.zeros((iters)), np.zeros((iters))
        maxReturn, minReturn = -np.inf, np.inf
        self.prepareCustomMetric()
        episodeLen = self.config["env"].get("max_length", MAX_EPISODE_LENGTH)
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
                testGain += reward
                testSteps += 1
                # Terminal condition for episode
                testDone = done
                if episodeLen > 0 and testSteps >= episodeLen:
                    testDone = True
                obs = proc(nextObs)
            # Processing metrics
            accRew[test] = testGain
            steps[test] = testSteps
            self.calculateCustomMetric()
            if testGain > maxReturn:
                maxReturn = testGain
            if testGain < minReturn:
                minReturn = testGain
        # calculate means and std
        meanAccReward, meanSteps = np.mean(accRew), np.mean(steps)
        stdMean, stdMeanSteps = np.std(accRew), np.std(steps)
        # Register to tb writer if available
        if self.tbw is not None:
            self.tbw.add_scalar("test/mean Return", meanAccReward)
            self.tbw.add_scalar("test/mean Steps", meanSteps)
            self.tbw.add_scalar("test/std Return", stdMean)
            self.tbw.add_scalar("test/std Steps", stdMeanSteps)
            self.tbw.add_scalar("test/max Return", maxReturn)
            self.tbw.add_scalar("test/min Return", minReturn)
        # Returning state
        self.prepareAfterTest()
        # Printing
        if prnt:
            print("Test results mean Return:{%.2f}, mean Steps:{%.2f}, std Return:{%.3f}, std Steps:{%.3f}" % (\
                meanAccReward, meanSteps, stdMean, stdMeanSteps))

        return {"mean_return": meanAccReward,
                "std_return": stdMean,
                "mean_steps": meanSteps, 
                "std_steps": stdMeanSteps,
                "custom": self.reportCustomMetric(),
                "max_return": maxReturn,
                "min_return": minReturn,
                }
        
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

    def prepareTest(self):
        """
            If the agent needs to prepare to save or load 
            anything before a test. Write it here
        """
        return None
    
    def prepareAfterTest(self):
        """
            If the agent needs to prepare to save or load 
            anything after a test. Write it here
        """
        return None
    
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
        return None

    def calculateCustomMetric(self):
        """
            From the state in test, and the self.envTest. Here custom code
            can be writen to calculate metrics differente from the test. No 
            variables are planned to be passed as argument more than self
        """
        return None

    def reportCustomMetric(self):
        """
            From the custom metric calculation, one can pass anything that can
            be saved to the test method. This will be stored on the test results
            dict.
        """
        return {}