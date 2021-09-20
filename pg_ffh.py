from rofl.algorithms.pg import train, config
from rofl.agents.pg import pgFFAgent 
from rofl.policies.pg import pgPolicy
from rofl.networks.pg import forestFireActorPG, forestFireBaseline, ffActor, ffBaseline
from rofl.envs import gymcaEnvMaker
from rofl.functions import getDevice, SummaryWriter, linearSchedule
from rofl.utils import seeder, Saver, expDir, saveConfig
from gym.spaces import Discrete, MultiDiscrete

EXP_NAME = "pg"
ENV_NAME = "ForestFireBulldozer-v1"
N_ACTIONS = 9, 2

#lr = linearSchedule(1e-3, 10**4, minValue = 1e-5)
#lr_b = linearSchedule(1e-3, 10**4, minValue = 1e-5)

# TODO support multidiscrete action space
# TODO change nets to support the one tensor obs
# TODO add to agent one tensor obs
# TODO add to policy one tensor obs

config["variables"] = []#[lr, lr_b]
config["env"]["name"] = ENV_NAME
config["env"]["obs_mode"] = "followGridPos"
config["env"]["reward_type"] = "hit_fire"
config["env"]["reward_move"] = 0.0
config["env"]["reward_hit"] = 1.0
config["env"]["reward_tree"] = 1.0
config["env"]["reward_fire"] = 0.0
config["env"]["n_col"] = 50
config["env"]["n_row"] = 50
config["env"]["wind_speed"] = 20
config["env"]["wind_direction"] = 160
config["env"]["obs_shape"] = (20, 20)
config["env"]["obs_channels"] = True
config["env"]["action_space"] = MultiDiscrete(N_ACTIONS)
config["policy"]["n_actions"] = N_ACTIONS
#config["policy"]["optimizer_args"] = {"weight_decay": 1e-5}
config["policy"]["net_hidden_1"] = 512
config["policy"]["learning_rate"] = 5e-5#lr
config["baseline"]["batch_minibatches"] = 20
config["baseline"]["learning_rate"] = 5e-5#lr_b
config["agent"]["lhist"] = 1
config["agent"]["clip_reward"] = 0.0
config["agent"]["no_op_start"] = 10
config["agent"]["scale_pos"] = True
config["train"]["epochs"] = 10**3
config["train"]["test_iters"] = 20
config["train"]["test_freq"] = 100
config["train"]["expected_perfomance"] = None
config["train"]["batch_size"] = 1000
config["train"]["batch_proportion"] = 1

device = getDevice()
seeder(8088, device)
#expdir, tbdir = expDir(EXP_NAME, ENV_NAME)
#saveConfig(config, expdir)

writer = None#SummaryWriter(tbdir)
saver = None#Saver(expdir)
actor = ffActor(config).to(device)
critic = ffBaseline(config).to(device)
policy = pgPolicy(config, actor, critic, tbw = writer)

envMaker = gymcaEnvMaker(config)
agent = pgFFAgent(config, policy, envMaker, tbw = writer)

if __name__ == "__main__":
    train(config, agent, policy, saver = saver)
    #writer.close()
