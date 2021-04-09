from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.functions.stop import testEvaluation, initResultDict
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
        "learning_rate":5e-5,
        "epsilon_start": 1,
        "epsilon_end": 0.1,
        "epsilon_test": 0.05,
        "epsilon_life": 10**6,
        "optimizer": "adam",
        "minibatch_size": 32,
        "freq_update_target":2500,
        "n_actions":18,
        "double": True,
        "evaluate_max_grad":True,
        "evaluate_mean_grad":True,
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
    },
}

def train(config:dict, agent:Agent, policy:Policy, saver = None):
    # Generate fixed trajectory
    sizeTrajectory = config["train"]["fixed_q_trajectory"]
    agent.tqdm = True
    trajectory = agent.getBatch(sizeTrajectory, 1.05)
    agent.fixedTrajectory = trajectory["st"]
    agent.reset()
    agent.tqdm = False
    # Fill memory replay
    sizeInitMemory = config["train"]["fill_memory"]
    I = tqdm(range(sizeInitMemory), desc="Filling memory replay")
    for _ in I:
        agent.step(randomPi = True)
    # Train the net
    trainResults = initResultDict()
    if saver is not None:
        saver.addObj(trainResults, "training_results")
        saver.start()
    miniBatchSize = config["train"]["mini_batch_size"]
    stepsPerEpoch = config["agent"]["steps_per_epoch"]
    freqTest = config["train"]["freq_test"]
    p = stepsPerEpoch / miniBatchSize
    epochs, stop = config["train"]["epochs"], False
    I = tqdm(range(epochs + 1), unit = "update", desc = "Training Policy")
    ## Train loop
    for epoch in I:
        # Train step
        miniBatch = agent.getBatch(miniBatchSize, p)
        policy.update(miniBatch)
        # Check for test
        if epoch % freqTest == 0:
            I.write("Testing ...")
            results, trainResults, stop = testEvaluation(config, agent, trainResults)
            I.write("Test results {}".format(results))
            
        # Stop condition
        if stop:
            if saver is not None:
                saver.saveAll()
            break
        # Check the saver status
        if saver is not None:
            saver.check()

    return trainResults
    
