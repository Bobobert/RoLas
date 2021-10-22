from typing import Union
from rofl import AgentMaster, Agent, Policy
from rofl.functions.const import *
from rofl.functions import testEvaluation, initResultDict, updateVar
from rofl.functions.torch import zeroGrad
from rofl.utils import Saver
from tqdm import tqdm

from rofl.config.defaults import network
baselineConf = network.copy()

algConfig = {
    'agent' :{
        'agentClass' : 'agentSync',
        'memory_size' : 10**3,
        'workers' : NCPUS,
        'workerClass' : 'a2cAgent',
        'clip_reward' : 1.0,
        'nstep' : 20,
    },

    'train' :{
        'batch_size' : -1,
        'batch_proportion' : 1.0,
        'test_freq' : 10**2,
        'modeGrad' : True, 
        # Modes: 
        # False multi agents for generating episode, 
        # True multi agent to generate grads
    },

    'policy' :{
        'policyClass' : 'a2cPolicy',
        'workerPolicyClass' : 'a2cWorkerPolicy',
        'entropy_bonus' : ENTROPY_LOSS,
        'n_actions' : None,
        'continuos' : False,
        'clip_grad' : 10.0,
        'minibatch_size' : MINIBATCH_SIZE,
        'loss_policy_const' : LOSS_POLICY_CONST,
        'loss_value_const' : LOSS_VALUES_CONST,
        'evaluate_tb_freq' : 10,
        'baseline' : baselineConf,
    }
}

def train(config:dict, agent:Union[Agent, AgentMaster], policy:Policy, saver: Saver):
    trainResults = initResultDict()
    saver.addObj(trainResults, 'train_results')
    saver.addObj(policy.actor, 'actor_net',
                isTorch = True, device = policy.device,
                key = 'mean_return')
    if getattr(policy, 'baseline', False) and not getattr(policy, 'actorHasCritic', False):
        if policy.baseline != None:
            saver.addObj(policy.baseline, 'baseline_net',
                    isTorch = True, device = policy.device,
                    key = 'mean_return')
    saver.start()

    freqTest = config['train']['test_freq']
    epochs, stop = config['train']['epochs'], False
    modeGrad = config['train']['modeGrad']
    I = tqdm(range(epochs + 1), unit = 'update', desc = 'Training Policy')
    agentMulti, multiEnv = isinstance(agent, AgentMaster), getattr(agent, 'isMultiEnv',False)
    
    try:
        ## Train loop
        zeroGrad(policy, True)
        for epoch in I:
            # Train step
            if not agentMulti:
                episode = agent.getEpisode()
                policy.update(episode)
            elif not modeGrad:
                episodes = agent.getEpisode()
                policy.update(*episodes)
            else:
                grads = agent.getGrads()
                policy.gradUpdate(*grads)
            if agentMulti and not multiEnv:
                agent.updateParams(*policy.getParams())
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
