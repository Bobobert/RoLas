"""
    Functions to manage actors and policies while developing or treating trajectories.
"""
from .dicts import addBootstrapArg
from rofl.utils.memory import episodicMemory, simpleMemory

def episodicRollout(agent, random = False):
    if agent._agentStep_ > 0:
        agent.reset()

    memory = episodicMemory(agent.config).reset()

    while not agent.done:
        memory.add(agent.fullStep(random = random))

    return memory.getEpisode()

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
    
    obsDict['return'] = obsDict['reward'] + agent.gamma * bootstrap # TODO: needs to be processed again?
    obsDict['bootstrapping'] = bootstrap
    obsDict['acuumulate_reward'] = obsDict['accumulate_reward'] + bootstrap
    return obsDict

def singlePathRollout(agent, maxLength = -1, memoryType = simpleMemory, 
                        reset = False, random = False, advantages = False):
    """
        Develops from the given state of the agent a rollout
        until reaches a terminal state or the length of the rollout
        is equal to maxLength if greater than 0.
        
        parameters
        ----------
        - agent: Agent type object
        - maxLength: int
        - memoryType: simpleMemory type
            Optional. If needed another type other than a simpleMemory
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
    if maxLength > agent.maxEpLength:
        maxLength = -1
        print('Warning: maxLength wont be reached, as the agents maxLength is lesser') # TODO: add debug level

    if reset and agent._agentStep_ > 0:
        agent.done = True
    memory = simpleMemory(agent.config)

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
