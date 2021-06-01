from rofl.algorithms.dqn import train, config
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice, SummaryWriter, linearSchedule
from rofl.utils import seeder, Saver, expDir, saveConfig

ENV_NAME = "breakout"
EXP_NAME = "dqn"
epsilon = linearSchedule(1.0, 25*10**4, minValue= 0.1)

config["variables"] = [epsilon]

config["env"]["name"] = ENV_NAME
config["env"]["frameskip"] = 4
config["env"]["obs_shape"] = (84,84)
config["env"]["max_steps_test"] = 10**4
config["env"]["atari"] = True
config["policy"]["n_actions"] = 4
config["policy"]["net_hidden_1"] = 512
config["policy"]["learning_rate"] = 5e-5
config["policy"]["epsilon"] = epsilon
config["policy"]["double"] = True
config["policy"]["freq_update_target"] = 10**4
config["agent"]["memory_prioritized"] = False
config["train"]["epochs"] = 1*10**6
config["train"]["fill_memory"] = 5*10**4
config["train"]["iters_test"] = 20
config["train"]["expected_performance"] = 300

device = getDevice()
seeder(8088, device)
expdir, tbdir = expDir(EXP_NAME, ENV_NAME)

writer = SummaryWriter(tbdir)
saver = Saver(expdir)
net = dqnAtari(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

envMaker = atariEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker, tbw = writer)
saveConfig(config, expdir)

def trainy():
    train(config, agent, policy, saver = saver)
#import cProfile
#cProfile.run("trainy()")
trainy()
writer.close()
