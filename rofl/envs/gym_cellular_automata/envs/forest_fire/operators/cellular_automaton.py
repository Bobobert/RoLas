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

    empty = CONFIG["cell_symbols"]["empty"]
    tree = CONFIG["cell_symbols"]["tree"]
    fire = CONFIG["cell_symbols"]["fire"]

    def __init__(self, grid_space=None, action_space=None, context_space=None):

        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def _op_update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = grid.copy()
        p_fire, p_tree = context

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):

                neighbors = neighborhood_at(grid, pos=(row, col), invariant=self.empty)

                if cell == self.tree and self.fire in neighbors:
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire

                elif cell == self.tree:
                    # Sample for lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1 - p_fire])[
                        0
                    ]
                    new_grid[row][col] = self.fire if strike else cell

                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1 - p_tree])[
                        0
                    ]
                    new_grid[row][col] = self.tree if growth else cell

                elif cell == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty

                else:
                    continue

        return new_grid, context

    def update(self, grid, action, context):
        throws = np.random.uniform(size = grid.shape)
        return _update_(grid, throws, context, self.fire, self.tree, self.empty), context
    
@numba.njit
def _update_(grid, throws, context, fire, tree, empty):
    pFire, pTree = context
    newGrid= np.zeros(grid.shape, dtype=grid.dtype)
    rows, cols = grid.shape

    for row in range(rows):
        for col in range(cols):
            fireArround, val = False, empty
            # Fire Around for Invariant Only
            for iRow in range(max(0,row - 1), min(rows, row+2)):
                for iCol in range(max(0,col - 1), min(cols, col+2)):
                    if (iRow != row or iCol != col) and (grid[iRow, iCol] == fire):
                        fireArround = True
            
            if (grid[row,col] == tree):
                if fireArround or (pFire > throws[row, col]):
                    # Burn tree to the ground
                    # Roll a dice for a lightning strike
                    val = fire
                else:
                    val = tree
            elif (grid[row, col] == empty) and (pTree > throws[row, col]):
                # Roll a dice for a growing bush
                val = tree
            elif grid[row, col] == fire:
                # Consume fire
                val = empty
            newGrid[row, col] = val

    return newGrid