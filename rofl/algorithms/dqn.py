from rofl import Agent, Policy
from rofl.functions.const import *
from rofl.functions.stop import testEvaluation, initResultDict
from rofl.functions.vars import updateVar
from tqdm import tqdm

dqnConfig = {
    "agent":{
        "lhist":LHIST,
        "memory_size":MEMORY_SIZE,
        "max_steps_test":10**4,
        "steps_per_epoch": 4,
        "clip_reward": 1.0,
        "scale_pos": True,
        "memory_prioritized": False,
    },
    "train":{
        "fill_memory":10**5,
        "fixed_q_trajectory":128,
        "mini_batch_size":32,
        "test_freq":5*10**4,

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
    }
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
    freqTest = config["train"]["test_freq"]
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
    