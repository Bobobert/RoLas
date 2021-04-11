from rofl.algorithms.dqn import train, config
from rofl.agents.drqn import drqnFFAgent
from rofl.policies.drqn import drqnPolicy
from rofl.networks.drqn import forestFireDRQNlstm
from rofl.envs.CA import forestFireEnvMaker
from rofl.functions import getDevice, SummaryWriter
from rofl.utils import seeder, Saver, expDir, saveConfig

EXP_NAME = "drqn"
ENV_NAME = "forest_fire_helicopter"

config["env"]["name"] = ENV_NAME
config["env"]["obs_mode"] = "followGridPos"
config["env"]["reward_type"] = "hit"
config["env"]["n_col"] = 100
config["env"]["n_row"] = 100
config["env"]["obs_shape"] = (52,52)
config["policy"]["recurrent_boot"] = 10
config["policy"]["n_actions"] = 9
config["agent"]["clip_reward"] = 0.0
config["agent"]["no_op_start"] = 10
config["train"]["epochs"] = 6*10**5
config["train"]["fill_memory"] = 10**3
config["train"]["iters_test"] = 20

device = getDevice()
seeder(8088, device)
expdir, tbdir = expDir(EXP_NAME, ENV_NAME)
saveConfig(config, expdir)

writer = SummaryWriter(tbdir)
saver = Saver(expdir)
net = forestFireDRQNlstm(config).to(device)
policy = drqnPolicy(config, net, tbw = writer)

envMaker = forestFireEnvMaker(config)
agent = drqnFFAgent(config, policy, envMaker, tbw = writer)

if __name__ == "__main__":
    train(config, agent, policy, saver = saver)
    writer.close()