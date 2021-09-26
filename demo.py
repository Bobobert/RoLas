from rofl.agents.pg import pgAgent00
from rofl.envs.gym import gymEnvMaker
from rofl.agents.multi import agentMaster, agentSync
from rofl.policies.base import dummyPolicy
from rofl.utils.memory import simpleMemory, episodicMemory
from tqdm import tqdm

config = {
            "env":{"name":"CartPole-v1", 
                    "seedTrain": 10, 
                    "seedTest": 20,
                    "max_length": -1}, 
            "agent":{"gamma":0.99, 
                    "gae": False,
                    "lambda":1.0,
                    "workers":-1,
                    "memory_size":10**4}}

maker = gymEnvMaker(config)
env, _ = maker()
policy = dummyPolicy(env)
"""
master = agentMaster(config, policy, maker, pgAgent)

master.reset()


for i in tqdm(range(10**4)):
    master.fullStep()

master.close()
"""
mem = episodicMemory(config)
agent = pgAgent00(config, policy, maker)

mem.reset()
for i in tqdm(range(10**4)):
    mem.add(agent.fullStep())

print(mem.sample(10))

epis = agent.getEpisode()

print(epis)
    
master = agentSync(config, policy, maker, pgAgent00)

master.reset()

print(master.getEpisodes())

master.close()