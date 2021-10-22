from rofl.agents.base import Agent
from rofl.agents.dqn import dqnAtariAgent, dqnCaAgent
from rofl.agents.pg import pgAgent
from rofl.agents.multi import agentSync
from rofl.agents.a2c import a2cAgent, caPgAgent

# TODO, include the agents to the config dict, use getattr to fetch them.
# for a function to play the agent! policies, and envMake should also comply with it
