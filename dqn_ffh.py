import rofl.algorithms as algorithms
from rofl.functions import getDevice, linearSchedule, createConfig
from rofl.utils import seeder, pathManager

from rofl.agents.dqn import dqnFFAgent, dqnFFAgent2
from rofl.policies.dqn import dqnPolicy
from rofl.networks.dqn import forestFireDQNth0, forestFireDQNv3, forestFireDQNVanilla, forestFireDQNv2, forestFireDQN

#  TODO; update this experiment!

envConfig = {
            'env':{
                    'envMaker' : 'forestFireEnvMaker',
                    'name': 'forest_fire_helicopter',
                    'n_row': 32,
                    'n_col': 32,
                    'p_tree': 0.01,
                    'p_fire': 0.005,
                    'ip_tree': 0.6,
                    'ip_fire': 0.0,
                    'obs_mode': 'followGridImg',
                    'obs_shape': (26,26,3),
                    'reward_type': 'hit_fire',
                    'reward_hit' : 0.5,
                    'reward_move' : 0.0,
                    'reward_tree' : 1.0,
                    'reward_fire' : 0.0,
                    'frameskip': 4,
                    'freeze':4,
                    'steps_termination' : 128,
                    'max_length': -1,
                    'warmup' : 'noop',
                }}

config = createConfig(envConfig, expName = 'dqn')
#lr = linearSchedule(1e-2, 10**6, minValue = 1e-4)
epsilon = linearSchedule(1.0, 5*10**5, minValue= 0.1)
config['variables'] = [epsilon]
 
 
'''config['policy']['net_hidden_1'] = 256 
config['policy']['net_hidden_2'] = 512 
config['policy']['net_hidden_3'] = 256 
config['policy']['net_hidden_4'] = 64 '''

config['agent']['lhist'] = 6
### NOT SUBJECT TO CHANGES ####

config['env']['n_col'] = 50
config['env']['n_row'] = 50
config['env']['ip_tree'] = 0.4
config['env']['p_tree'] = 0.05
config['env']['p_fire'] = 0.005
config['env']['freeze'] = 9
config['env']['steps_termination'] = 500
config['policy']['n_actions'] = 9
config['policy']['epsilon'] = epsilon
config['policy']['learning_rate'] = 5e-5
config['agent']['memory_size'] = 10**6
config['agent']['clip_reward'] = 0.0
config['agent']['no_op_start'] = 20
config['train']['epochs'] = 10**6
config['train']['fill_memory'] = 50000
config['train']['test_iters'] = 20

device = getDevice()
seeder(8088, device)
expdir, tbdir = expDir(EXP_NAME, ENV_NAME)

writer = SummaryWriter(tbdir)
saver = Saver(expdir)
net = forestFireDQNv3(config).to(device)
policy = dqnPolicy(config, net, tbw = writer)

agent = dqnFFAgent(config, policy, envMaker, tbw = writer)

if __name__ == '__main__':
    saveConfig(config, expdir)
    train(config, agent, policy, saver = saver)
    writer.close()
