from tqdm import tqdm

from rofl import AgentType, PolicyType
from rofl.functions.const import LHIST, MEMORY_SIZE
from rofl.functions import testEvaluation, initResultDict
from rofl.utils import Saver, updateVar

algConfig = {
    'env':{
        'reward_function' : None,
    },
    'agent':{
        'lhist' : LHIST,
        'channels' : 1,
        'use_context' : True,
        'memory_size' : MEMORY_SIZE,
        'max_steps_test' : 10**4,
        'steps_per_epoch' : 4,
        'clip_reward' : 1.0,
        'scale_pos' : True,
        'memory_prioritized' : False,
    },
    'train':{
        'fill_memory' : 10**5,
        'fixed_q_trajectory' : 128,
        'mini_batch_size' : 32,
        'test_freq' : 5*10**4,

    },
    'policy':{
        'policyClass' : 'DqnPolicy',
        'epsilon_start': 1.0,
        'epsilon_end' : 0.1,
        'epsilon_life' : 25 * 10**4,
        'epsilon_test' : 0.05,
        'minibatch_size' : 32,
        'freq_update_target' : 10**4, # was 2500 before
        'n_actions' : 18,
        'double' : True,
    }
}

def fillRandomMemoryReplay(config:dict, agent:AgentType):
    # Fill memory replay
    sizeInitMemory = config['train']['fill_memory']
    I = tqdm(range(sizeInitMemory), desc='Filling memory replay', unit='envStep')
    for _ in I:
        agent.memory.add(agent.fullStep(random=True))

def fillFixedTrajectory(config:dict, agent:AgentType, device):
    # Generate fixed trajectory
    sizeTrajectory = config['train']['fixed_q_trajectory']
    agent.memory.reset()
    trajectory = agent.getBatch(sizeTrajectory, random=True, progBar=True, device=device)
    agent.fixedTrajectory = trajectory['observation']
    agent.memory.reset()
    agent.reset()

def train(config:dict, agent:AgentType, policy:PolicyType, saver: Saver):
    policy.test = True
    fillFixedTrajectory(config, agent, policy.device)
    fillRandomMemoryReplay(config, agent)
    policy.train = True
    # Train the net
    ## Init results and saver
    trainResults = initResultDict()
    saver.addObj(trainResults, 'training_results')
    saver.addObj(policy.dqnOnline,'online_net',
                isTorch=True, device=policy.device,
                key='mean_return')
    saver.start()

    miniBatchSize = config['policy']['minibatch_size']
    stepsPerEpoch = config['agent']['steps_per_epoch']
    freqTest = config['train']['test_freq']
    p = miniBatchSize / stepsPerEpoch
    epochs, stop = config['train']['epochs'], False
    I = tqdm(range(epochs + 1), unit='update', desc='Training Policy')

    try:
    ## Train loop
        for epoch in I:
            # Train step
            miniBatch = agent.getBatch(miniBatchSize, p, device=policy.device)
            policy.update(miniBatch)
            updateVar(config)
            # Check for test
            if epoch % freqTest == 0:
                I.write('Testing ...')
                results, trainResults, stop = testEvaluation(config, agent, trainResults)
                I.write('Test results {}'.format(results))
                # Check the saver status
                if stop == '':
                    saver.check(results)
                else:
                    I.write(stop)
                    break
    except KeyboardInterrupt:
        print("Keyboard termination . . .")
        
    saver.saveAll(results)
    return trainResults
        