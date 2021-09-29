from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.config.defaults import network
from rofl.functions.stop import testEvaluation, initResultDict
from rofl.functions.vars import updateVar
from rofl.utils import Saver
from tqdm import tqdm

baselineConf = network.copy()
baselineConf['minibatch_size'] = MINIBATCH_SIZE
baselineConf['batch_minibatches'] = 10

algConfig = {
    "agent" :{
        "clip_reward" : 1.0,
    },
    "train" :{
        "batch_size" : -1,
        "batch_proportion" : 1.0,
        "test_freq" : 10**2,
    },
    "policy" :{
        "entropy_bonus" : ENTROPY_LOSS,
        "n_actions" : 18,
        "max_div_kl" : MAX_DKL,
        "surrogate_epsilon" : EPS_SURROGATE,
        "loss_policy_const" : LOSS_POLICY_CONST,
        "loss_value_const" : LOSS_VALUES_CONST,
        'evaluate_tb_freq' : 10**3,
        "baseline" : baselineConf,
    }
}

def train(config:dict, agent:Agent, policy:Policy, saver: Saver):
    # Train the net
    ## Init results and saver
    trainResults = initResultDict()
    saver.addObj(trainResults, "training_results")
    saver.addObj(policy.actor, "actor_net",
                isTorch = True, device = policy.device,
                key = 'mean_return')
    if getattr(policy, 'baseline', False):
        saver.addObj(policy.baseline, "baseline_net",
                isTorch = True, device = policy.device,
                key = 'mean_return')
    saver.start()

    batchSize, p = config["train"]["batch_size"], config["train"]["batch_proportion"]
    freqTest = config["train"]["test_freq"]
    epochs, stop = config["train"]["epochs"], False
    I = tqdm(range(epochs + 1), unit = "update", desc = "Training Policy")
    try:
        ## Train loop
        for epoch in I:
            # Train step
            episode = agent.getEpisode()
            policy.update(episode)
            updateVar(config)
            # Check for test
            if epoch % freqTest == 0:
                I.write("Testing ...")
                results, trainResults, stop = testEvaluation(config, agent, trainResults)
                I.write("Test results {}".format(results))
                # Stop condition
                if not stop:
                    saver.check(results)
                else:
                    saver.saveAll(results)
                    return trainResults
        saver.saveAll(results)
        return trainResults
    except KeyboardInterrupt:
        print("Keyboard termination. Saving all objects in Saver")
        saver.saveAll(results)
