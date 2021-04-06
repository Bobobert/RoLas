from rofl.algorithms.dqn import train, config
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice

config["env"]["name"] = "seaquest"
config["policy"]["n_actions"] = 6
config["train"]["epochs"] = 100
config["env"]["obs_shape"] = (84,84)

device = getDevice()

net = dqnAtari(config).to(device)
policy = dqnPolicy(config, net)

envMaker = atariEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker)

train(config, agent, policy)
