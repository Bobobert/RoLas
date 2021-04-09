from rofl.algorithms.dqn import train, config
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import forestFireDQN
from rofl.envs.CA import forestFireEnvMaker
from rofl.functions import getDevice, SummaryWriter
from rofl.utils import seeder, Saver, expDir, saveConfig

EXP_NAME = "dqn"
ENV_NAME = "forest_fire_helicopter"

config["env"]["name"] = ENV_NAME
config["env"]["obs_mode"] = "followGridImg"
config["env"]["reward_type"] = "hit"
config["policy"]["n_actions"] = 9
config["agent"]["clip_reward"] = 0.0
config["agent"]["no_op_start"] = 10
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
net = forestFireDQN(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

envMaker = forestFireEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker, tbw = writer)

if __name__ == "__main__":
    train(config, agent, policy, saver = saver)
    writer.close()
