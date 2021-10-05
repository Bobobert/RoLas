from rofl.functions.const import GAMMA, LAMDA_GAE, OPTIMIZER_DEF, OPTIMIZER_LR_DEF, TRAIN_SEED, TEST_SEED

agent = {
    'agentClass' : None, # TODO, perhaps to validate this keys after createConfig
    'id' : 0,
    'gamma' : GAMMA,
    'lambda' : LAMDA_GAE,
    'gae' : False,
}

train = {
    'epochs' : 10**2,
    'test_freq' : 10,
    'test_iters' : 20,
    'expected_perfomance' : None,
    'max_performance' : None,
    'max_time' : None,
    'max_steps_per_test' : -1,
}

network = {
    'networkClass' : None,
    'linear_1' : 512,
    'optimizer' : OPTIMIZER_DEF,
    'learning_rate' : OPTIMIZER_LR_DEF,
    'optimizer_args' : {},
    }

policy = {
    'policyClass' : None,
    'evaluate_tb_freq' : 15,
    'evaluate_max_grad' : True,
    'evaluate_mean_grad' : True,
    'clip_grad' : 0,
    'network' : network.copy(),
}

env = {
    'envMaker' : 'gymEnvMaker', # TODO
    'name' : None,
    'warmup' : None,
    'warmup_min_steps' : 0,
    'warmup_max_steps' : 30,
    'obs_shape' : None,
    'obs_mode' : None,
    'max_length' : 10**3,
    'seedTrain' : TRAIN_SEED,
    'seedTest' : TEST_SEED,
}

config = {
    'env' : env,
    'agent' : agent,
    'policy' : policy,
    'train' : train,
    'variables' : [],
    'algorithm' : 'unknown',
    'version' : 'v0.2.0',
    'seed' : 42,
}
