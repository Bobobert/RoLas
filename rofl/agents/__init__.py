# TODO, include the agents to the config dict, use getattr to fetch them.
# for a function to play the agent! policies, and envMake should also comply with it

from rofl.agents.base import BaseAgent
from rofl.agents.dqn import DqnAtariAgent, DqnCAAgent
from rofl.agents.pg import PgAgent
from rofl.agents.multi import AgentSync
from rofl.agents.a2c import A2CAgent, PgCAAgent
