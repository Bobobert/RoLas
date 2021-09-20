from rofl.algorithms.dqn import train, config
from rofl.envs.CA import forestFireEnvMaker
from rofl.functions import getDevice, SummaryWriter, linearSchedule
from rofl.utils import seeder, Saver, expDir, saveConfig

from rofl.agents.dqn import dqnFFAgent, dqnFFAgent2
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import forestFireDQNth0, forestFireDQNv3, forestFireDQNVanilla, forestFireDQNv2, forestFireDQN

EXP_NAME = "dqn"
ENV_NAME = "forest_fire_helicopter_2"

#lr = linearSchedule(1e-2, 10**6, minValue = 1e-4)
epsilon = linearSchedule(1.0, 5*10**5, minValue= 0.1)
config["variables"] = [epsilon]

config["env"]["name"] = ENV_NAME
config["env"]["obs_mode"] = "followGridPos"
config["env"]["obs_shape"] = (26,26,3)
config["env"]["reward_type"] = "hit_fire"
config["env"]["reward_move"] = 0.0
config["env"]["reward_hit"] = 0.5
config["env"]["reward_tree"] = 1.0
config["env"]["reward_fire"] = 0.0
 
 
"""config["policy"]["net_hidden_1"] = 256 
config["policy"]["net_hidden_2"] = 512 
config["policy"]["net_hidden_3"] = 256 
config["policy"]["net_hidden_4"] = 64 """

config["agent"]["lhist"] = 6
### NOT SUBJECT TO CHANGES ####

config["env"]["n_col"] = 50
config["env"]["n_row"] = 50
config["env"]["ip_tree"] = 0.4
config["env"]["p_tree"] = 0.05
config["env"]["p_fire"] = 0.005
config["env"]["freeze"] = 9
config["env"]["steps_termination"] = 500
config["policy"]["n_actions"] = 9
config["policy"]["epsilon"] = epsilon
config["policy"]["learning_rate"] = 5e-5
config["agent"]["memory_size"] = 10**6
config["agent"]["clip_reward"] = 0.0
config["agent"]["no_op_start"] = 20
config["train"]["epochs"] = 10**6
config["train"]["fill_memory"] = 50000
config["train"]["test_iters"] = 20

device = getDevice()
seeder(8088, device)
expdir, tbdir = expDir(EXP_NAME, ENV_NAME)

writer = SummaryWriter(tbdir)
saver = Saver(expdir)
net = forestFireDQNv3(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

envMaker = forestFireEnvMaker(config)
agent = dqnFFAgent(config, policy, envMaker, tbw = writer)

if __name__ == "__main__":
    saveConfig(config, expdir)
    train(config, agent, policy, saver = saver)
    writer.close()
