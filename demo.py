from rofl.agents.pg import pgAgent
from rofl.envs.gym import gymEnvMaker
from rofl.agents.multi import agentMaster
from rofl.policies.dummy import dummyPolicy

config = {
            "env":{"name":"CartPole-v1", 
                    "seedTrain": 10, 
                    "seedTest": 20,
                    "max_length": -1}, 
            "agent":{"gamma":1.0, 
                    "gae": False,
                    "lambda":1.0}}

maker = gymEnvMaker(config)
env, _ = maker()
policy = dummyPolicy(env)

master = agentMaster(config, policy, maker, pgAgent)

master.reset()

from tqdm import tqdm
for i in tqdm(range(10**4)):
    master.fullStep()

master.close()

agent = pgAgent(config, policy, maker)
r = []
for i in tqdm(range(10**4)):
    r.append(agent.fullStep())
    
