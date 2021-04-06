import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

import gym
from gym import spaces
from gym.utils import seeding
import numba
import math

from .operators import (
    ForestFireCellularAutomaton,
    ForestFireModifier,
    ForestFireCoordinator,
)
from .utils.config import get_forest_fire_config_dict
from .utils.render import plot_grid, add_helicopter
from .utils.initializers import init_bulldozer, generate_wind_kernel

CONFIG = get_forest_fire_config_dict()

CELL_STATES = CONFIG["cell_states"]

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

FIRES = CONFIG["ca_params"]["fires"]
INIT_P_TREE = CONFIG["ca_params"]["init_p_tree"]
WIND_SPEED = CONFIG["ca_params"]["wind_speed"]
WIND_DIRECTION = CONFIG["ca_params"]["wind_direction"]
WIND_C1 = CONFIG["ca_params"]["wind_c1"]

EFFECTS = CONFIG["effects"]

TIMES = CONFIG["mdp_internal_times"]
MAX_TIME = max(TIMES.values())

ACTION_MIN = CONFIG["actions"]["min"]
ACTION_MAX = CONFIG["actions"]["max"]

# spaces.Box requires typing for discrete values
CELL_TYPE = CONFIG["cell_type"]
ACTION_TYPE = CONFIG["action_type"]

OBS_MODE = CONFIG["obs_mode"]["mode"]
OBS_SHAPE = (CONFIG["obs_mode"]["shapeRow"], CONFIG["obs_mode"]["shapeCol"])

# ------------ Forest Fire Environment

# NEW CONTEXT
# bulldozer's position, internal mdp's time, alive status

class BulldozerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    empty = CONFIG["cell_symbols"]["empty"]
    tree = CONFIG["cell_symbols"]["tree"]
    fire = CONFIG["cell_symbols"]["fire"]
    burnt = CONFIG["cell_symbols"]["burnt"]

    pos_space = spaces.MultiDiscrete([ROW, COL])
    times_space = spaces.Discrete(MAX_TIME + 1)
    alive_space = spaces.Discrete(2)

    context_space = spaces.Tuple((pos_space, times_space, alive_space))
    grid_space = spaces.Box(0, CELL_STATES - 1, shape=(ROW, COL), dtype=CELL_TYPE)

    action_space = spaces.Box(ACTION_MIN, ACTION_MAX, shape=tuple(), dtype=ACTION_TYPE)
    observation_space = spaces.Tuple((grid_space, context_space))

    wind = generate_wind_kernel(WIND_DIRECTION, WIND_SPEED, WIND_C1)
    last_counts = None

    obs_mode = OBS_MODE
    obsShape = OBS_SHAPE

    reward_mode = CONFIG["reward_mode"]

    def __init__(self):

        self.cellular_automaton = ForestFireCellularAutomaton(
            grid_space=self.grid_space,
            wind=self.wind,
            action_space=self.action_space,
        )

        self.modifier = ForestFireModifier(
            EFFECTS,
            TIMES,
            endCells = {self.fire},
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.context_space,
        )

        self.coordinator = ForestFireCoordinator(
            self.cellular_automaton, self.modifier, mdp_time = TIMES,
            context_space = self.context_space
        )

        self.reward_per_empty = CONFIG["rewards"]["per_empty"]
        self.reward_per_tree = CONFIG["rewards"]["per_tree"]
        self.reward_per_fire = CONFIG["rewards"]["per_fire"]
        self.reward_per_burnt = CONFIG["rewards"]["per_burnt"]
        self.reward_cut = CONFIG["rewards"]["cut"]
        self.reward_alive = CONFIG["rewards"]["alive"]
        print("New bulldozer env created, kernel\n {}".format(self.wind))

    def reset(self):
        self.grid, pos = init_bulldozer(self.grid_space,
                                        self.empty,
                                        self.tree,
                                        self.fire,
                                        INIT_P_TREE,
                                        fires = FIRES)
        # NEW CONTEXT
        # bulldozer's position, internal mdp's time, alive status
        self.context = pos, 0, True
        self.coordinator.last_lattice_update = 0
        self.modifier.contact = False
        self.last_counts = None
        self.init_trees = Counter(self.grid.flatten().tolist())[self.tree]
        #obs = self.grid, self.context
        obs = self.obsGrid()

        return obs

    def step(self, action):
        done = self._is_done()

        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            self.grid = new_grid
            self.context = new_context

        #obs = self.grid, self.context
        obs = self.obsGrid()
        reward = self._award() if not done else 0.0
        info = self._report()

        return obs, reward, done, info

    def _award(self):
        _, _, alive = self.context
        dict_counts = Counter(self.grid.flatten().tolist())
        reward = 0
        
        if self.reward_mode == "hit":
            new_burnt = dict_counts[self.burnt] - self.last_counts[self.burnt]
            reward += new_burnt * self.reward_per_burnt
            reward += self.reward_per_tree * dict_counts[self.tree] if not alive else self.reward_alive
        elif self.reward_mode == "ratio":
            reward += self.reward_per_burnt * (dict_counts[self.burnt] / self.init_trees)
            reward += self.reward_per_tree * (dict_counts[self.tree] / self.init_trees) if not alive else self.reward_alive

        reward += self.modifier.hit * self.reward_cut

        self.last_counts = dict_counts   
        return reward

    def _is_done(self):
        _, _, alive = self.context
        if self.last_counts is None:
            self.last_counts = Counter(self.grid.flatten().tolist())
        if self.last_counts[self.fire] == 0 or not alive:
            return True
        return False

    def _report(self):
        return {"hit": self.modifier.hit, "alive": not self.modifier.contact}

    def render(self, mode="human"):
        pos, _, _ = self.context

        figure = add_helicopter(plot_grid(self.grid), pos)
        plt.show()

        return figure
    # MOD TO DQN IMAGE OUTPUT
    def obsGrid(self):
        if self.obs_mode == "follow":
            return self.followGrid()
        elif self.obs_mode == "static":
            return self.staticGrid()
        else:
            raise KeyError("Observation mode not available")

    @staticmethod
    @numba.njit
    def quickGrid(grid:np.ndarray, fire, tree, empty, burnt):
        """
        Hard coded values for the cell type in gridImg
        """
        FIRE = 130
        TREE = 50
        BURNT = 90
        EMPTY = 0
        img = np.zeros(grid.shape, dtype=np.uint8)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                currentCell, val = grid[row,col], 0
                if currentCell == fire:
                    val = FIRE
                elif currentCell == tree:
                    val = TREE
                elif currentCell == burnt:
                    val = BURNT
                else:
                    val = EMPTY
                img[row, col] = val
        return img
    
    def getImgGrid(self):
        """Returns a ndarray uint8 with a imagen representation 
        of all the grid"""
        gridShape = self.grid.shape
 
        rowBig = gridShape[0] > self.obsShape[0]
        colBig = gridShape[1] > self.obsShape[1]
        
        img = self.quickGrid(self.grid, 
                                self.fire, self.tree, self.empty, self.burnt)

        (row, col), _, _ = self.context
        img[row, col] = 255 # Agent Helicopter is the brigthest spot on the plane

        if not rowBig and colBig:
            # Padding rows
            diff = self.obsShape[0] - gridShape[0]
            newGrid = np.zeros((self.obsShape[0], gridShape[1]), dtype=np.uint8)
            newGrid[diff:,:] = img
            img = newGrid
        elif rowBig and not colBig:
            # Padding cols
            diff = self.obsShape[1] - gridShape[1]
            newGrid = np.zeros((gridShape[0], self.obsShape[1]), dtype=np.uint8)
            newGrid[:,diff:] = img
            img = newGrid
        elif not rowBig and not colBig:
            # Padding all the image
            imgBig = np.zeros(self.obsShape, dtype=np.uint8)
            r = (self.obsShape[0] - gridShape[0]) // 2
            c = (self.obsShape[1] - gridShape[1]) // 2
            imgBig[r:r+gridShape[0],c:c+gridShape[1]] = img
            img = imgBig

        return img

    def followGrid(self):
        """
        Follows the agent through the grid
        """
        gridShape = self.grid.shape[:2]
        
        img = self.getImgGrid()
        (row_pos, col_pos), _, _ = self.context
        if 0 > (row_pos - math.floor(self.obsShape[0] / 2)):
            rMin, rMax = 0, self.obsShape[0]
        elif (row_pos + math.ceil(self.obsShape[0] / 2)) >= gridShape[0]:
            rMax = gridShape[0]
            rMin = rMax - self.obsShape[0]
        else:
            rMin = row_pos - math.floor(self.obsShape[0] / 2)
            rMax = rMin + self.obsShape[0]

        if 0 > (col_pos - math.floor(self.obsShape[1] / 2)):
            cMin, cMax = 0, self.obsShape[1]
        elif (col_pos + math.ceil(self.obsShape[1] / 2)) >= gridShape[1]:
            cMax = gridShape[1]
            cMin = cMax - self.obsShape[1]
        else:
            cMin = col_pos - math.floor(self.obsShape[1] / 2)
            cMax = cMin + self.obsShape[1]

        return img[rMin:rMax, cMin:cMax]

    def staticGrid(self):
        img = self.getImgGrid()
        (row_pos, col_pos), _, _ = self.context
        gridShape = self.grid.shape
        #sectorR = math.ceil(gridShape[0] / (self.obsShape[0] - 2))
        #sectorC = math.ceil(gridShape[1] / (self.obsShape[1] - 2))
        agentRSec = row_pos // (self.obsShape[0] - 2)
        agentCSec = col_pos // (self.obsShape[1] - 2)
        rMax = min(gridShape[0], (self.obsShape[0] - 2) * (agentRSec + 1))
        cMax = min(gridShape[1], (self.obsShape[1] - 2) * (agentCSec + 1))
        rMin = max(0, rMax - self.obsShape[0] + 2)
        cMin = max(0, cMax - self.obsShape[1] + 2)

        newImg = np.zeros(self.obsShape,dtype=np.uint8)
        newImg[1:-1,1:-1] = img[rMin:rMax, cMin:cMax]
        Fire = 130

        if not rMax == gridShape[0]:
            # Down side
            mid = math.floor(self.obsShape[0] / 2)
            newImg[-1,mid-1:mid+2] = Fire
        if not rMin == 0:
            # Upper side
            mid = math.floor(self.obsShape[0] / 2)
            newImg[0, mid-1:mid+2] = Fire
        if not cMax == gridShape[1]:
            # Rigth side
            mid = math.floor(self.obsShape[1] / 2)
            newImg[mid-1:mid+2,-1] = Fire
        if not cMin == 0:
            # left side
            mid = math.floor(self.obsShape[1] / 2)
            newImg[mid-1:mid+2,0] = Fire
        return newImg

    def seed(self, s):
        self.cellular_automaton.rg = np.random.Generator(np.random.SFC64(s))
        return [s]