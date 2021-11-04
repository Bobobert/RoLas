from typing import Tuple
from rofl.functions.functions import multiplyIter

def rewardFuncTemplate(agent, obs, reward, info, done):
    return reward

def getCells(agent) -> Tuple[int, int, int, int]:
    env =  agent.env
    grid = env.grid
    counts = env.count_cells(grid)
    tot = multiplyIter(grid.shape)
    fire = counts[env._fire]
    tree = counts[env._tree]
    burnt = counts[env._burned]

    return tot, fire, tree, burnt

def hitFire(agent, obs, reward, info, done):
    tot, fire, tree, burnt = getCells(agent)
    reward_ = tree / tot * 1.0
    reward_ += fire / tot * 0.0
    