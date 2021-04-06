from rofl.algorithms.dqn import train, config
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice

config["env"]["name"] = "seaquest"
config["env"]["frameskip"] = 4
config["env"]["obs_shape"] = (84,84)
config["policy"]["n_actions"] = 6
config["train"]["epochs"] = 10**5
config["train"]["fill_memory"] = 10**3
config["train"]["iters_test"] = 5

device = getDevice()

net = dqnAtari(config).to(device)
policy = dqnPolicy(config, net)

envMaker = atariEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker)


def trainy():
    train(config, agent, policy)
#import cProfile
#cProfile.run("trainy()")
trainy()
