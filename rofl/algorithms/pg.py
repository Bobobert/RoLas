from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.functions import testEvaluation, initResultDict, updateVar
from rofl.utils import Saver
from tqdm import tqdm

from rofl.config.defaults import network
baselineConf = network.copy()

algConfig = {
    'agent' :{
        'agentClass' : 'pgAgent',
        'memory_size' : 10 ** 3,
        'clip_reward' : 1.0,
        'nstep' : -1,
    },

    'train' :{
        'batch_size' : -1,
        'batch_proportion' : 1.0,
        'test_freq' : 10**2,
    },

    'policy' :{
        'policyClass' : 'pgPolicy',
        'entropy_bonus' : ENTROPY_LOSS,
        'n_actions' : None,
        'continuos' : False,
        'minibatch_size' : MINIBATCH_SIZE,
        'loss_policy_const' : LOSS_POLICY_CONST,
        'loss_value_const' : LOSS_VALUES_CONST,
        'evaluate_tb_freq' : 10**3,
        'baseline' : baselineConf,
    }
}

def train(config:dict, agent:Agent, policy:Policy, saver: Saver):
    # Train the net
    ## Init results and saver
    trainResults = initResultDict()
    saver.addObj(trainResults, 'training_results')
    saver.addObj(policy.actor, 'actor_net',
                isTorch = True, device = policy.device,
                key = 'mean_return')
    if getattr(policy, 'baseline', False) and not getattr(policy, 'actorHasCritic', False):
        if policy.baseline != None:
            saver.addObj(policy.baseline, 'baseline_net',
                    isTorch = True, device = policy.device,
                    key = 'mean_return')
    saver.start()

    batchSize, p = config['train']['batch_size'], config['train']['batch_proportion']
    freqTest = config['train']['test_freq']
    epochs, stop = config['train']['epochs'], False
    I = tqdm(range(epochs + 1), unit = 'update', desc = 'Training Policy')
    
    try:
        ## Train loop
        for epoch in I:
            # Train step
            episode = agent.getEpisode()
            policy.update(episode)
            updateVar(config)
            # Check for test
            if epoch % freqTest == 0:
                I.write('Testing ...')
                results, trainResults, stop = testEvaluation(config, agent, trainResults)
                I.write('Test results {}'.format(results))
                # Stop condition
                if stop == '':
                    saver.check(results)
                else:
                    I.write(stop)
                    break
    except KeyboardInterrupt:
        print('Keyboard termination. . .')

    saver.saveAll(results)
    return trainResults
