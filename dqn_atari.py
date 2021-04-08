from rofl.algorithms.dqn import train, config
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice, SummaryWriter
from rofl.utils import seeder, Saver, expDir, saveConfig

ENV_NAME = "seaquest"
EXP_NAME = "dqn"

config["env"]["name"] = ENV_NAME
config["env"]["frameskip"] = 4
config["env"]["obs_shape"] = (84,84)
config["env"]["max_steps_test"] = 10**4
config["policy"]["n_actions"] = 6
config["train"]["epochs"] = 5*10**5
config["train"]["fill_memory"] = 10**3
config["train"]["iters_test"] = 20
config["train"]["max_performance"] = 1000

device = getDevice()
seeder(8088, device)
expdir, tbdir = expDir(EXP_NAME, ENV_NAME)
saveConfig(config, expdir)

writer = SummaryWriter(tbdir)
saver = Saver(expdir)
net = dqnAtari(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

envMaker = atariEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker, tbw = writer)

def trainy():
    train(config, agent, policy, saver = saver)
#import cProfile
#cProfile.run("trainy()")
trainy()
writer.close()
