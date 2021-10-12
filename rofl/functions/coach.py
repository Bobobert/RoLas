"""
    Functions to manage actors and policies while developing or treating trajectories.
"""
from .dicts import addBootstrapArg
from rofl.functions.const import *
from rofl.functions.functions import no_grad
from rofl.utils.memory import episodicMemory

@no_grad()
def calcBootstrap(agent):
    """
        Calculates the current value if able of the ongoing agent state.

    """
    return agent.processReward(agent.policy.getValue(agent.lastObs, agent.lastAction))

def prepareBootstrapping(agent, obsDict):
    """
        Process the last obsDict to have a done flag and bootstraps
        from the policy if able.
    """
    
    obsDict = addBootstrapArg(obsDict)
    
    if obsDict['done'] == True:
        return obsDict

    obsDict['bootstrapping'] = bootstrap = calcBootstrap(agent)
    obsDict['acuumulate_reward'] = obsDict['accumulate_reward'] + bootstrap
    obsDict['done'] = True
    return obsDict

def singlePathRollout(agent, maxLength = -1, memory: episodicMemory = None,
                        reset: bool = False, random: bool = False, forceLen: bool = False):
    """
        Develops from the given state of the agent a rollout
        until reaches a terminal state or the length of the rollout
        is equal to maxLength if greater than 0.
        
        parameters
        ----------
        - agent: Agent type object
        - maxLength: int
            Default -1, so it does a whole episode. Else, the number of steps to do 
            in this rollout.
        - memory: episodicMemory type
            Optional. If want to use another memory. By default creates a episodicMemory
        - reset: bool
            Optional. Default False. To ensure a fresh state is generated for the environment
        - random: bool
            Optional. If the actions will be total random or will follow the policy
        - forceLen: bool
            If maxLength > 0 and this is True, then the number of steps of maxLength are forced
            into the memory. Else, the rollout stop whenever a normal terminal condition is reached,
            ie, when terminal state.
        returns
        -------
        Memory

    """
    if forceLen and maxLength < 1:
        raise ValueError('When forcing length the maximum should be set a positive quantity!')

    if maxLength > agent.maxEpLen and not forceLen:
        maxLength = -1
        print('Warning: maxLength wont be reached, as the agents maxLen (%d) is lesser' % agent.maxEpLen) # TODO: add debug level

    if reset and not agent._reseted: # avoiding reseting agent more than one per sample
        agent.done = True

    if memory is None:
        memory = episodicMemory(agent.config)
        memory.reset()

    stepsDone = 0
    while True:
        obsDict = agent.fullStep(random = random)
        memory.add(obsDict)
        stepsDone += 1

        if obsDict['done']:
            if not forceLen:
                break
            elif stepsDone == maxLength:
                break

        elif maxLength > 0:
            if stepsDone == maxLength:
                prepareBootstrapping(agent, obsDict)
                memory.resolveReturns()
                break
    
    return memory
