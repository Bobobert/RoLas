"""
    Functions to manage actors and policies while developing or treating trajectories.
"""
from .dicts import addBootstrapArg, obsDict
from rofl.functions.const import *
from rofl.utils.memory import episodicMemory, simpleMemory

def episodicRollout(agent, *additionalKeys, random = False, device = DEVICE_DEFT):
    if agent._agentStep_ > 0:
        agent.done = True
    

    memory = episodicMemory(agent.config, *additionalKeys)
    memory.reset()

    while True:
        obsDict_ = agent.fullStep(random = random)
        memory.add(obsDict_)
        if obsDict_['done']: break

    return memory.getEpisode(device = device)

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

    bootstrap = calcBootstrap(agent)
    
    obsDict['return'] += agent.gamma * bootstrap # TODO: needs to be processed again?
    obsDict['bootstrapping'] = bootstrap
    obsDict['acuumulate_reward'] = obsDict['accumulate_reward'] + bootstrap
    return obsDict

def singlePathRollout(agent, maxLength = -1, memory = None, 
                        reset = False, random = False, advantages = False):
    """
        Develops from the given state of the agent a rollout
        until reaches a terminal state or the length of the rollout
        is equal to maxLength if greater than 0.
        
        parameters
        ----------
        - agent: Agent type object
        - maxLength: int
        - memory: simpleMemory obj
            Optional. If want to use another memory. By default creates a simpleMemory
        -reset: bool
            Optional. To ensure a fresh state is generated for the environment
        - random: bool
            Optional. If the actions will be total random or will follow the policy
        - advantage: bool
            If true, each step will add to the observation a value calculated, in order to 
            estimate the advantage for the given observation. Else, will just calculate the bootstrap
            for the last state reached if its not terminal.
        returns
        -------
        Memory

    """
    # TODO; make sure this has the n-step sampling logic it should have!
    if maxLength > agent.maxEpLength:
        maxLength = -1
        print('Warning: maxLength wont be reached, as the agents maxLength is lesser') # TODO: add debug level

    if reset and agent._agentStep_ > 0:
        agent.done = True

    if memory is None:
        keys = [('G_t', F_TDTYPE_DEFT), ('bootstrapping', F_TDTYPE_DEFT)] if advantages else []
        memory = simpleMemory(agent.config, ('return', F_TDTYPE_DEFT), *keys)
        memory.reset() 

    stepsInit, endBySteps, done, lastGt = agent._agentStep_, False, False, 0.0

    if advantages:
        lastGt = calcBootstrap(agent)

    while True:
        obsDict = agent.fullStep(random = random)
        if advantages:
            prepareBootstrapping(agent, obsDict)
            obsDict['advantage'] = obsDict['return'] - lastGt
            obsDict['G_t'] = lastGt
            lastGt = obsDict['bootstrapping']
        if maxLength > 0:
            endBySteps = True if obsDict['step'] - stepsInit >= maxLength else False
        if obsDict['done'] or endBySteps:
            prepareBootstrapping(agent, obsDict) if not advantages else None
            obsDict['done'], done = True, True
        memory.add(obsDict)
        if done:
            break

    return memory
