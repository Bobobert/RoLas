import numpy as np
from gym import spaces

from rofl.envs.gym_cellular_automata import Operator
from ..utils.neighbors import are_my_neighbors_a_boundary
from ..utils.config import get_forest_fire_config_dict

CONFIG = get_forest_fire_config_dict()

ACTION_UP_LEFT = CONFIG["actions"]["up_left"]
ACTION_UP = CONFIG["actions"]["up"]
ACTION_UP_RIGHT = CONFIG["actions"]["up_right"]

ACTION_LEFT = CONFIG["actions"]["left"]
ACTION_NOT_MOVE = CONFIG["actions"]["not_move"]
ACTION_RIGHT = CONFIG["actions"]["right"]

ACTION_DOWN_LEFT = CONFIG["actions"]["down_left"]
ACTION_DOWN = CONFIG["actions"]["down"]
ACTION_DOWN_RIGHT = CONFIG["actions"]["down_right"]

ACTION_CUT = CONFIG["actions"]["cut"]

UP_SET = {ACTION_UP_LEFT, ACTION_UP, ACTION_UP_RIGHT}
DOWN_SET = {ACTION_DOWN_LEFT, ACTION_DOWN, ACTION_DOWN_RIGHT}

LEFT_SET = {ACTION_UP_LEFT, ACTION_LEFT, ACTION_DOWN_LEFT}
RIGHT_SET = {ACTION_UP_RIGHT, ACTION_RIGHT, ACTION_DOWN_RIGHT}

ACTION_MIN = CONFIG["actions"]["min"]
ACTION_MAX = CONFIG["actions"]["max"]

ACTION_TYPE = CONFIG["action_type"]

# ------------ Forest Fire Modifier


class ForestFireModifier(Operator):
    is_composition = False
    hit = False
    contact = False # As stop condition

    def __init__(self, effects, mdp_time, endCells = None, grid_space=None, action_space=None, context_space=None):

        self.effects = effects
        self.times = mdp_time

        if action_space is None:
            action_space = spaces.Box(
                ACTION_MIN, ACTION_MAX, shape=tuple(), dtype=ACTION_TYPE
            )

        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space
        self.endCells = endCells

    def update(self, grid, action, context):
        pos, internal_time, alive = context
        new_pos, cut = self.bulldozer_move(grid, action, pos)
        row, col = new_pos

        self.hit = False

        # Apply effect only if the action was to change - move does not applies effects
        if cut:
            for symbol in self.effects:
                if grid[row, col] == symbol:
                    grid[row, col] = self.effects[symbol]
                    self.hit = True
        
        # Adding to the internal times
        # It adds time whenever the bulldozer moves or not
        internal_time += self.times["bulldozer_move" if not cut else "bulldozer_cut"]

        # Check if bulldozer is not in the cells type end cell
        grid, context = self.bulldozer_status(grid, (new_pos, internal_time, alive))

        return grid, context

    def bulldozer_status(self, grid, context):
        pos, internal_time, alive = context
        if self.endCells is not None:
            row, col = pos
            if grid[row, col] in self.endCells:
                alive = False
                self.contact = True
        context = (pos, internal_time, alive)
        return grid, context

    def bulldozer_move(self, grid, action, pos):
        action = int(action)
        # Just a version consider when the action is to cut
        if action == ACTION_CUT:
            return pos, True
        
        bounds = grid.shape
        row, col = pos
        if action in UP_SET:
            row += -1
        elif action in DOWN_SET:
            row += 1
        if (row < 0) or (row >= bounds[0]):
            row = pos[0]
        if action in RIGHT_SET:
            col += 1
        elif action in LEFT_SET:
            col += -1
        if (col < 0) or (col >= bounds[1]):
            col = pos[1]
    
        return np.array([row, col]), False
