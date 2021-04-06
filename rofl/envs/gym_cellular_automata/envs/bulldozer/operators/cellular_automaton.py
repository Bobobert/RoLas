import numpy as np
from gym import spaces

from rofl.envs.gym_cellular_automata import Operator

from ..utils.neighbors import neighborhood_at
from ..utils.config import get_forest_fire_config_dict

CONFIG = get_forest_fire_config_dict()

import numba

# ------------ Forest Fire Cellular Automaton
class ForestFireCellularAutomaton(Operator):
    is_composition = False
    
    rg = np.random.Generator(np.random.SFC64())
    empty = CONFIG["cell_symbols"]["empty"]
    tree = CONFIG["cell_symbols"]["tree"]
    fire = CONFIG["cell_symbols"]["fire"]
    burnt = CONFIG["cell_symbols"]["burnt"]

    def __init__(self, grid_space=None, wind=None, action_space=None, context_space=None):

        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.grid_space = grid_space
        self.wind = wind
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        throws = np.random.uniform(size = grid.shape)
        return _update_(grid, self.wind, throws, \
            self.fire, self.tree, self.empty, self.burnt), context
    
@numba.njit
def _update_(grid:np.ndarray, wind:np.ndarray, throws:np.ndarray, fire, tree, empty, burnt):

    newGrid = np.zeros(grid.shape, dtype=grid.dtype)
    probsGrid = np.zeros(grid.shape, dtype=np.float32)
    rows, cols = grid.shape

    POS = [1,2,0]
    # Sweeping cells from op grid for wind probs
    for row in range(rows):
        for col in range(cols):
            # If cell is fire then apply wind kernel for its propagation
            if grid[row, col] == fire:
                # Between possible neighborhood
                for iRow in range(max(0,row - 1), min(rows, row+2)):
                    for iCol in range(max(0,col - 1), min(cols, col+2)):
                        # Add the probability to the neighborhood cells
                        dRow = iRow - row
                        dCol = iCol - col
                        probsGrid[iRow, iCol] += wind[POS[dRow], POS[dCol]]

    # Applying transitions
    for row in range(rows):
        for col in range(cols):
            cell = grid[row, col]
            if cell == fire:
                # Consume fire
                val = burnt
            elif (cell == tree) and (probsGrid[row, col] >= throws[row, col]):
                # Add new fire
                val = fire
            else:
                # empty or burnt don't change state ever again
                val = cell
            newGrid[row, col] = val

    return newGrid