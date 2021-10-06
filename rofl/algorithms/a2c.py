from rofl import Agent, Policy
from rofl.config.defaults import network
from rofl.functions.const import *
from rofl.functions.stop import testEvaluation, initResultDict
from rofl.functions.vars import updateVar
from rofl.utils import Saver
from tqdm import tqdm


baselineConf = network.copy()
baselineConf['minibatch_size'] = MINIBATCH_SIZE
baselineConf['batch_minibatches'] = 10

algConfig = {
    'agent' :{
        'agentClass' : 'agentSync',
        'memory_size' : 10**4,
        'workers' : NCPUS,
        'workerClass' : 'a2cAgent',
        'clip_reward' : 1.0,
        'nstep' : -1,
    },

    'train' :{
        'batch_size' : -1,
        'batch_proportion' : 1.0,
        'test_freq' : 10**2,
    },

    'policy' :{
        'policyClass' : 'a2cPolicy',
        'workerPolicyClass' : 'a2cWorkerPolicy',
        'entropy_bonus' : ENTROPY_LOSS,
        'n_actions' : None,
        'continuos' : False,
        'shared_memory' : True,
        'clip_grad' : 10.0,
        'minibatch_size' : MINIBATCH_SIZE,
        'max_div_kl' : MAX_DKL,
        'surrogate_epsilon' : EPS_SURROGATE,
        'loss_policy_const' : LOSS_POLICY_CONST,
        'loss_value_const' : LOSS_VALUES_CONST,
        'evaluate_tb_freq' : 10**3,
        'baseline' : baselineConf,
    }
}

def train(config:dict, agent:Agent, policy:Policy, saver: Saver):
    trainResults = initResultDict()
    saver.addObj(trainResults, 'train_results')
    saver.addObj(policy.actor, 'actor_net',
                isTorch = True, device = policy.device,
                key = 'mean_return')
    if getattr(policy, 'baseline', False):
        if policy.baseline != None:
            saver.addObj(policy.baseline, 'baseline_net',
                    isTorch = True, device = policy.device,
                    key = 'mean_return')
    saver.start()

    freqTest = config['train']['test_freq']
    epochs, stop = config['train']['epochs'], False
    I = tqdm(range(epochs + 1), unit = 'update', desc = 'Training Policy')
    
    try:
        ## Train loop
        for epoch in I:
            # Train step
            episodes = agent.getEpisode()
            policy.update(*episodes)
            params = policy.getParams()
            agent.updateParams(*params)
            #otherParams = agent.getParams()[0]
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
