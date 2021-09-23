from rofl.functions.const import *
from rofl.functions.functions import nprnd, runningMean, multiplyIter
from rofl.agents.base import Agent
from rofl.agents.multi import agentMaster
from rofl.functions.torch import updateNet
from rofl.functions.exploratory import qUCBAction

class dqnRolloutAgent(Agent):
    name = "dqn-rollout agent base"

    def initAgent(self, heuristic = None, **kwargs):
        self.heuristic = heuristic
        if heuristic is None:
            raise ValueError("Heuristic needs to be not a NoneType")
        config = self.config
        self.ucbC = config["agent"].get("ubc_c", 1.0)
        self.rolloutDepth = config["agent"].get("rollout_depth", 1)
        self.rolloutLength = config["agent"].get("rollout_length", 10)
        self.rolloutSamples = config["agent"].get("rollout_samples", 30)
        self.heuristicArgs = config["agent"].get("heuristic_args", {})
        self.nActions = config["policy"]["n_actions"]

    def startRollout(self, state, action, nt):
        
        rt = 0.0 # mean R_t
        gs = 0.0

        for i in range(self.rolloutSamples):
            # Reset state
            self.loadState(state)
            obs0 = self.envStep(action)
            # mean Rt
            rt = runningMean(rt, obs0["reward"], i)
            rs, gm, keep, d = obs0["reward"], self.gamma, True, 0

            if obs0["done"]: 
                keep = False

            while d < self.rolloutLength and keep:
                # Do the UCB actions
                if d < self.rolloutDepth:
                    # take UCB action
                    At = qUCBAction(self, self._envStep_, nt[self._envStep_], self.ucbC)
                    nt[self._envStep_, At] += 1
                else:
                    # Use heuristic
                    At = self.heuristicAction()
                obsDict = self.envStep(At)
                rs += gm * obsDict["reward"]
                d += 1
                gm *= self.gamma
                # Stop condition for the actual rollout
                if obsDict["done"]:
                    keep = False
            # Bootstrap last value with dqn
            if keep:
                qvalues = self.getQvalues(self.lastObs)
                rs += gm * qvalues.max().item()
            gs = runningMean(gs, rs, i)
            # TODO insert time / resources limit stop condition here

        return gs, rt, nt, action
    
    def heuristicAction(self):
        return self.heuristic(self.env, self.lastObs, **self.heuristicArgs)

    def getQvalues(self, obs):
        return self.policy.dqnOnline.forward(obs).squeeze()

    def updateDQN(self, netParameters):
        updateNet(self.policy.dqnOnline, netParameters)

from rofl.agents.dqn import prepare4Ratio, calRatio, reportRatio

class dqnRollFFAgent(dqnRolloutAgent):
    name = "DQNrollout ForestFire Agentv0"

    def initAgent(self, **kwargs):
        super().initAgent(**kwargs)
        config = self.config

        from rofl.utils.memory import dqnMemory
        self.memory = dqnMemory(config)

        obsShape, lhist  = config["env"]["obs_shape"], config["agent"]["lhist"]
        self.obsShape = (lhist, multiplyIter(obsShape) + 3)
        self.fixedTrajectory = None
        self.frameStack = np.zeros(self.obsShape, dtype = F_NDTYPE_DEFT)

    def processObs(self, obs, reset: bool = False):
        """
            If the agent needs to process the observation of the
            environment. Write it here
        """
        frame, pos, tm = obs["frame"], obs["position"], obs.get("time", 0)
        if reset:
            self.frameStack.fill(0)
        else:
            self.frameStack = np.roll(self.frameStack, 1, axis = 0)

        self.frameStack[0,:-3] = frame.flatten() / 255
        self.frameStack[0,-3:-1] = pos
        self.frameStack[0, -1] = tm

        return torch.from_numpy(self.frameStack).unsqueeze(0)

    def prepareTest(self):
        prepare4Ratio(self)

    def calculateCustomMetric(self, env, reward, done):
        calRatio(self, env)

    def reportCustomMetric(self):
        return reportRatio(self)

    def currentState(self):
        state = super().currentState()
        # for FFCA env
        env = self.env
        state["env_state"] = (env.grid.copy(), env.pos_row, env.pos_col, env.total_hits, env.total_reward, 
            env.remaining_moves, env.steps_to_termination, env.first_termination, env.total_burned,
            env.terminated, env.steps, env.last_move)
        return state

    def loadState(self, newState):
        super().loadState(newState)
        env = self.env
        env.grid, env.pos_row, env.pos_col, env.total_hits, env.total_reward, \
            env.remaining_moves, env.steps_to_termination, env.first_termination, env.total_burned, \
            env.terminated, env.steps, env.last_move = newState["env_state"]

    def createNT(self):
        """
            Returns a Tensor that has a shape 
            [Episode length, Number actions]
        """
        c = self.config
        l = c["env"].get("max_length", 1024)
        n = c["policy"]["n_actions"]
        return torch.zeros((l, n), dtype = I_TDTYPE_DEFT)

class multiAgentRollout(agentMaster):

    def reset(self):
        super().reset()
        self.nActions = self.config["policy"]["n_actions"]

    def rollout(self, state, nt):
        """
            parameters
            ----------
            state: dict
                An agent state from the method .currentState, should
                contain enough information to load the environment state
            nt: Tensor
                The number of times an action has been seen at a time t

            returns
            --------
            - action from rollout
            - nt modified
            - gs per action, or the Q(s,a) value from rollout
            - rt per action, the average R_t for the action observed from
                the rollouts
        """
        results, i = [], 0
        free, working, _ = self.listWorkers()
        AS = [i for i in range(self.nActions)]
        # Pool for the task
        while AS != []:
            if free == []:
                [r], free, working = self.asyncResolve(working)
                results.append(r)
            w = free.pop()
            w.ref = w().startRollout.remote(state, AS.pop(), nt.clone())
            working.append(w)
        results += self.syncResolve(working)

        # Analyse results
        nt = -self.nActions * nt 
        max_gs, max_action = -np.inf, None
        gs_actions = torch.zeros((self.nActions), dtype = F_TDTYPE_DEFT)
        rt_actions = torch.zeros((self.nActions), dtype = F_TDTYPE_DEFT)
        for result in results:
            gs, rt, nt_A, action = result
            nt += nt_A
            if gs > max_gs:
                max_gs = gs
                max_action = [action]
            if gs == max_gs:
                max_action.append(action)
            gs_actions[action] = gs
            rt_actions[action] = rt
        if len(max_action) > 1:
            max_action = nprnd.choice(max_action)
        else:
            max_action = max_action[-1]
        
        return max_action, nt, gs_actions, rt_actions
