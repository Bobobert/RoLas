from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.functions.stop import testEvaluation, initResultDict
from rofl.functions.vars import updateVar
from tqdm import tqdm

config = {
    "agent":{
        "lhist":LHIST,
        "memory_size":MEMORY_SIZE,
        "gamma":GAMMA,
        "max_steps_test":10**4,
        "steps_per_epoch": 4,
        "clip_reward": 1.0,
        "no_op_start": 30,
        "scale_pos": True,
        "memory_prioritized": False,
        "gae": 1.0,
        "lambda": 1.0,
    },
    "train":{
        "fill_memory":10**5,
        "fixed_q_trajectory":128,
        "epochs":10**6,
        "mini_batch_size":32,
        "freq_test":5*10**4,
        "iters_test":20,
        "expected_perfomance": None,
        "max_performance": None,
        "max_time": None,
    },
    "policy":{
        "learning_rate":OPTIMIZER_LR_DEF,
        "epsilon": 1.0,
        "epsilon_test": 0.05,
        "optimizer": OPTIMIZER_DEF,
        "minibatch_size": 32,
        "freq_update_target":2500,
        "n_actions":18,
        "net_hidden_1": 328,
        "double": True,
        "evaluate_max_grad":True,
        "evaluate_mean_grad":True,
        "recurrent_unit":"lstm",
        "recurrent_layers":1,
        "recurrent_hidden_size":328,
        "recurrent_units":1,
        "recurrent_boot":10,
        "clip_grad": 0.0,
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
        "steps_termination" : 128,
    },
}

def fillRandomMemoryReplay(config:dict, agent:Agent):
    # Fill memory replay
    sizeInitMemory = config["train"]["fill_memory"]
    I = tqdm(range(sizeInitMemory), desc="Filling memory replay")
    for _ in I:
        expDict = agent.fullStep(random = True)
        agent.memory.add(expDict)

def fillFixedTrajectory(config, agent, device):
    # Generate fixed trajectory
    sizeTrajectory = config["train"]["fixed_q_trajectory"]
    trajectory = agent.getBatch(sizeTrajectory, progBar = True, device = device)
    agent.fixedTrajectory = trajectory["observation"]
    agent.memory.reset()
    agent.reset()

def train(config:dict, agent:Agent, policy:Policy, saver = None):
    agent.memory.reset()
    fillFixedTrajectory(config, agent, policy.device)
    fillRandomMemoryReplay(config, agent)

    # Train the net
    ## Init results and saver
    trainResults = initResultDict()
    if saver is not None:
        saver.addObj(trainResults, "training_results")
        saver.addObj(policy.dqnOnline,"online_net",
                    isTorch = True, device = policy.device)
        saver.start()
    def saverAll():
        if saver is not None:
            saver.saveAll()

    miniBatchSize = config["policy"]["minibatch_size"]
    stepsPerEpoch = config["agent"]["steps_per_epoch"]
    freqTest = config["train"]["freq_test"]
    p = miniBatchSize / stepsPerEpoch
    epochs, stop = config["train"]["epochs"], False
    I = tqdm(range(epochs + 1), unit = "update", desc = "Training Policy")
    ## Train loop
    for epoch in I:
        # Train step
        miniBatch = agent.getBatch(miniBatchSize, p, device = policy.device)
        policy.update(miniBatch)
        updateVar(config)
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
    