from rofl.policies.base import BasePolicy
from rofl.agents.multi import AgentMaster


class Rollout(BasePolicy):

    def getAction(self, observation, **kwargs):
        return super().getAction(observation, **kwargs)

