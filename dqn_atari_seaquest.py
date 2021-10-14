from rofl.algorithms.dqn import train, algConfig
from rofl.agents.dqn import dqnAtariAgent
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import dqnAtari
from rofl.envs.gym import atariEnvMaker
from rofl.functions import getDevice, createConfig, linearSchedule
from rofl.utils import seeder, pathManager

device = getDevice()
seeder(8080, device)

def setUp():
    epsilon = linearSchedule(1.0, 0.1, 25*10**4)

    envConfig = {
                "env":{
                        "name": 'seaquest',
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
                        "max_length": -1,
                        'warmup' : 'noop',
                    }}

    config = createConfig(algConfig, envConfig, expName = 'dqn')
    config["variables"] = [epsilon]
    config['agent']['memory_size'] = 7*10**5
    config["policy"]["n_actions"] = 6
    config["policy"]["net_hidden_1"] = 512
    config["policy"]["learning_rate"] = 1e-5
    config["policy"]["epsilon"] = epsilon
    config["policy"]["double"] = True
    config["policy"]["freq_update_target"] = 10**4
    config["train"]["epochs"] = 2*10**6
    config["train"]["fill_memory"] = 5*10**4
    config['train']['test_freq'] = 5*10**4
    config["train"]["max_steps_per_test"] = 10**4
    config["train"]["test_iters"] = 20
    config["train"]["expected_performance"] = 1500

    manager = pathManager(config, dummy = False)
    writer = manager.startTBW()
    net = dqnAtari(config).to(device)
    policy = dqnPolicy(config, net, tbw = writer)

    envMaker = atariEnvMaker(config)
    agent = dqnAtariAgent(config, policy, envMaker, tbw = writer)

    manager.saveConfig()
    return config, agent, policy, manager

if __name__ == '__main__':
    config, agent, policy, manager = setUp()
    train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
    #import cProfile
    #cProfile.run("trainy()")
    #trainy()
