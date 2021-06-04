from rofl.algorithms.pg import train, config
from rofl.agents.pg import pgAgent 
from rofl.policies.pg import pgPolicy
from rofl.networks.pg import dcontrolActorPG, ccBaseline
from rofl.envs.gym import gymEnvMaker
from rofl.functions import getDevice, SummaryWriter
from rofl.utils import seeder, Saver, expDir, saveConfig
from gym.spaces import Discrete

EXP_NAME = "pg"
ENV_NAME = "CartPole-v0"
N_ACTIONS = 2

config["env"]["name"] = ENV_NAME
config["env"]["action_space"] = Discrete(N_ACTIONS)
config["env"]["max_length"] = 500
config["agent"]["lhist"] = 4
config["agent"]["clip_reward"] = 0.0
config["agent"]["no_op_start"] = 0
config["policy"]["n_inputs"] = 4
config["policy"]["n_actions"] = N_ACTIONS
config["policy"]["clip_grad"] = 0.0
config["policy"]["entropy_bonus"] = 0.0
config["train"]["epochs"] = 10**4
config["train"]["freq_test"] = 500
config["train"]["batch_size"] = 500
config["train"]["batch_proportion"] = 1
config["train"]["iters_test"] = 20
config["train"]["max_performance"] = 200

device = getDevice()
seeder(8088, device)
#expdir, tbdir = expDir(EXP_NAME, ENV_NAME)
#saveConfig(config, expdir)

writer = None#SummaryWriter(tbdir)
saver = None#Saver(expdir)
actor = dcontrolActorPG(config).to(device)
critic = ccBaseline(config).to(device)
policy = pgPolicy(config, actor, critic, tbw = writer)

envMaker = gymEnvMaker(config)
agent = pgAgent(config, policy, envMaker, tbw = writer)

if __name__ == "__main__":
    train(config, agent, policy, saver = saver)
    writer.close()
