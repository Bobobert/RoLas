from rofl.functions.const import ALPHA
from rofl.config.types import AgentType


class RolloutAgent(AgentType):
    name = "Rollout simple agent"
    def __init__(self, config, envMaker) -> None:
        self.config = config
        self.env, self.envSeed = envMaker(config['agent']['seedTrain'])

## better to make a wrapper for the agent class that one will work with to 
# have access to the already done methods

def BasicActionGen(actionStart, lookahead, depth):
    yield [actionStart]


class RolloutAgentWrap:
    def __init__(self, agent, heuristic) -> None:
        self.agent = agent
        self.heuristic = heuristic 
        if heuristic is None:
            # try ?
            self.heuristic = self.rdnAction

    def rollout(self, state, observation, actionSeq: list, depth:int = 10, gamma: float = 10.0,
                    randomState = None):
        """
        Parameters
        ----------
        state:
            Point of start for the rollout
        obsevation: 
            Observation from which to quick start the rollout
        actionSeq:
            list of the defined sequence of actions to execute. These are expected to be selected or
            as a product of a given lookahead. This list should not be empty.
        depth: int
            The amount of steps expected to do in this rollout. The path could end sooner if a terminal
            observation is reached.
        gamma: float
            Default the gamma of the agent wrapped. If given a value in [0, 1] the rollout reward is modified
            by this amount instead.
        randomState:
            An object to modify or keep track of the random state; if used, on the heuristic

        Returns
        --------
        actionStart:
            The first action took in this rollout path
        rolloutReward:
            The discounted accumulated reward for the path
        actionSequence:
            All the actions developed in the rollout path

         
        actionGen:
            Alledgely the from a set of n available actions, the problem could required to choose
            more one of the sequence of actions instead of an action alone. An actionGen should express
            this sequences from the starting actions. Meaning is another iterator to call this method in
            a recursive from (??).
        
        """
        #assert lookahead <= depth, 'Lookahead cannot be greater than the depth of the rollout'
        if len(actionSeq) > depth:
            print('Action sequence is equal to the depth, no other actions will be taken!')

        rolloutReward = 0
        actionStart, rollActionSeq = actionSeq[0], []
        
        gamma = gamma if gamma <= 1.0 and gamma >= 0.0 else self.gamma
        runnigGamma = 1.0
        obs = observation
        heuristic, env = self.heuristic, self.env

        self.loadEnvState(state)

        for i in range(depth):
            if i < len(actionSeq):
                action = actionSeq[i]
            else:
                action = heuristic(env, obs, randomState)

            rollActionSeq.append(action)
            obsDict = self.envStep(action)
            rolloutReward += runnigGamma * obsDict['reward']
            runnigGamma *= gamma
            obs = obsDict['next_observation']
            if obsDict['done']:
                break
        
        # Add here if need a modification to the rolloutReward in here
        
        rolloutReward += runnigGamma * 0.0

        return actionStart, rolloutReward, rollActionSeq

    def loadEnvState(self, state):
        self.prevEnvState, self.prevLastObs = self.getEnvState(self.env)
        self.setEnvState(state)

    def getEnvState(self):
        raise NotImplementedError
        return (), self.lastObs

    def setEnvState(self, state):
        raise NotImplementedError
        return
        