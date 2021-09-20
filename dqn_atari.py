from rofl.algorithms.dqn import train, dqnConfig
from rofl.algorithms import createConfig
from rofl.agents.newdqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice, SummaryWriter, linearSchedule
from rofl.utils import seeder, Saver, expDir, saveConfig

ENV_NAME = "seaquest"
EXP_NAME = "dqn"
epsilon = linearSchedule(1.0, 25*10**4, minValue = 0.1)

envConfig = {
            "env":{
                    "name": ENV_NAME,
                    "atari": True,
                    "n_row": 32,
                    "n_col": 32,
                    "p_tree": 0.01,
                    "p_fire": 0.005,
                    "ip_tree": 0.6,
                    "ip_fire": 0.0,
                    "obs_mode": "followGridImg",
                    "obs_shape": (84,84),
                    "reward_type": "hit",
                    "frameskip": 4,
                    "freeze":4,
                    "steps_termination" : 128,
                    "max_length": 2500,
                }}

config = createConfig(dqnConfig, envConfig)
config["variables"] = [epsilon]
config["policy"]["n_actions"] = 4
config["policy"]["net_hidden_1"] = 512
config["policy"]["learning_rate"] = 5e-5
config["policy"]["epsilon"] = epsilon
config["policy"]["double"] = True
config["policy"]["freq_update_target"] = 10**4
config["train"]["epochs"] = 1*10**5
config["train"]["fill_memory"] = 1*10**4
config["train"]["max_steps_per_test"] = 10**4
config["train"]["test_iters"] = 20
config["train"]["expected_performance"] = 300

device = getDevice()
seeder(8088, device)
#expdir, tbdir = expDir(EXP_NAME, ENV_NAME)

writer = None#SummaryWriter(tbdir)
saver = None#Saver(expdir)
net = dqnAtari(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

envMaker = atariEnvMaker(config)
agent = dqnAtariAgent(config, policy, envMaker, tbw = writer)
#saveConfig(config, expdir)

if __name__ == '__main__':
    #def trainy():
    train(config, agent, policy, saver = saver)
    #import cProfile
    #cProfile.run("trainy()")
    #trainy()
    #writer.close()
