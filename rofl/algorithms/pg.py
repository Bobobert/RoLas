from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.functions.stop import testEvaluation, initResultDict
from tqdm import tqdm

config = {
    "agent":{
        "lhist":LHIST,
        "memory_size":MEMORY_SIZE,
        "gamma":GAMMA,
        "lambda": LAMDA_GAE,
        "gae": False,
        "max_steps_test":10**4,
        "clip_reward": 1.0,
        "no_op_start": 30,
    },
    "train":{
        "n_workers": 1,
        "epochs": 10**3,
        "batch_size": -1,
        "batch_proportion" : 1.0,
        "freq_test": 10**2,
        "iters_test": 20,
        "expected_perfomance": None,
        "max_performance": None,
        "max_time": None,
    },
    "policy":{
        "learning_rate":OPTIMIZER_LR_DEF,
        "optimizer": OPTIMIZER_DEF,
        "entropy_bonus" : 1e-2,
        "n_actions": 18,
        "evaluate_max_grad":True,
        "evaluate_mean_grad":True,
        "recurrent_unit":"lstm",
        "recurrent_layers":1,
        "recurrent_hidden_size":328,
        "recurrent_units":1,
        "recurrent_boot":10,
        "clip_grad": 0.0,
    },
    "baseline":{
        "learning_rate":OPTIMIZER_LR_DEF,
        "optimizer":OPTIMIZER_DEF,
        "minibatch_size" : MINIBATCH_SIZE,
        "batch_minibatches" : 10,
    },
    "env":{
        "name":"forestFire",
        "atari": False,
        "n_row": 32,
        "n_col": 32,
        "p_tree": 0.01,
        "p_fire": 0.005,
        "ip_tree": 0.6,
        "ip_fire": 0.0,
        "obs_mode": "followGridImg",
        "obs_shape": (32,32),
        "reward_type": "hit",
        "frameskip": 4,
        "max_length": -1,
        "seedTrain" : 10,
        "seedTest": 1,
        "freeze":4,
    },
}

def train(config:dict, agent:Agent, policy:Policy, saver = None):
    # Train the net
    ## Init results and saver
    trainResults = initResultDict()
    if saver is not None:
        saver.addObj(trainResults, "training_results")
        saver.addObj(policy.actor,"actor_net",
                    isTorch = True, device = policy.device)
        if policy.baseline is not None:
            saver.addObj(policy.baseline, "baseline_net",
                    isTorch = True, device = policy.device)
        saver.start()
    def saverAll():
        if saver is not None:
            saver.saveAll()
    batchSize, p = config["train"]["batch_size"], config["train"]["batch_proportion"]
    freqTest = config["train"]["freq_test"]
    epochs, stop = config["train"]["epochs"], False
    I = tqdm(range(epochs + 1), unit = "update", desc = "Training Policy")
    ## Train loop
    for epoch in I:
        # Train step
        batch = agent.getBatch(batchSize, p)
        policy.update(batch)
        # Check for test
        if epoch % freqTest == 0:
            I.write("Testing ...")
            results, trainResults, stop = testEvaluation(config, agent, trainResults)
            I.write("Test results {}".format(results))
        # Stop condition
        if stop:
            saverAll()
            break
        # Check the saver status
        if saver is not None:
            saver.check()
    saverAll()
    return trainResults