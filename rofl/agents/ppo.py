from rofl.functions.functions import no_grad
from rofl.agents.a2c import a2cAgent
from rofl.utils.policies import getBaselines

class ppoAgent(a2cAgent):
    name = 'ppo agent'
    
    def getEpisode(self, random=False, device=None):
        episode = super().getEpisode(random=random, device=device)
        with no_grad():
            baselines = getBaselines(self.policy, episode['observation'])
        advantage = episode['return'] - baselines
        episode['advantage'] = advantage.detach_()
        return episode
