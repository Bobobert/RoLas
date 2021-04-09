# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:57:09 2020

@author: ebecerra
"""
# Rob Was Here

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from .forest_fire import ForestFire
from gym.spaces import Discrete
from gym import Env
import numba
#import gif

#gif.options.matplotlib["dpi"] = 150

FIRE = 130
TREE = 50
EMPTY = 0

# Implements positions and movement and fire embedding
class Helicopter(ForestFire):
    """
    Helicopter class
    simulates a Helicopter over a Firest Forest Automaton

    Superclass for EnvForestFire
    For more please check the documentation of EnvForestFire

    Examples
    --------
    >>> helicopter = Helicopter()
    >>> helicopter.render()
    >>> helicopter.movement_actions
    >>> helicopter.new_pos(7)
    >>> helicopter.render()
    """
    
    def __init__(self, init_pos_row = None, init_pos_col = None,
                 n_row = 16, n_col = 16, p_tree=0.100, p_fire=0.001,
                 forest_mode = 'stochastic', custom_grid = None,
                 force_fire = True, boundary='invariant',
                 tree = '|', empty = '.', fire = '*', rock = '#', lake = 'O',
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None,
                 is_copy=False):
        self.movement_actions = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        self.action_set = self.movement_actions
        # Forest Fire Parameters
        kw_params={'n_row':n_row, 'n_col':n_col, 'p_tree':p_tree, 'p_fire':p_fire, 'forest_mode':forest_mode,
                 'custom_grid':custom_grid, 'force_fire':force_fire, 'boundary':boundary,
                 'tree':tree, 'empty':empty, 'fire':fire, 'rock':rock, 'lake':lake,
                 'ip_tree':ip_tree, 'ip_empty':ip_empty, 'ip_fire':ip_fire, 'ip_rock':ip_rock, 'ip_lake':ip_lake}
        ForestFire.__init__(self,**kw_params)
        self.init_pos_row = init_pos_row
        self.init_pos_col = init_pos_col
        # Helicopter attributes
        if init_pos_row is None:
            # Start aprox in the middle
            self.pos_row = self.n_row//2
        else:
            self.pos_row = init_pos_row
        if init_pos_col is None:
            # Start aprox in the middle
            self.pos_col = self.n_col//2
        else:
            self.pos_col = init_pos_col
        self.last_move = False
        # RWH. New added variables for new functions.
        self.checkpoints = []
        self.checkpoint_counter = 0
        self.frames = [] # Potential high memory usage

    def new_pos(self, movement):
        """
        Changed on 25/01/2021
        Now to 
        [5 1 6]
        [2 0 3]
        [7 4 8]
        
        to support both 8C and 4C movements.
        """
        a = (self.pos_row, self.pos_col)
        iRow, iCol = 0,0
        if movement == 0:
            None
        if movement in {5,1,6}:
            iRow += -1
        elif movement in {7,4,8}:
            iRow += 1
        if movement in {5,2,7}:
            iCol += -1
        elif movement in {6,3,8}:
            iCol += 1
        self.pos_row = max(0, min(self.grid.shape[0] - 1, self.pos_row + iRow))
        self.pos_col = max(0, min(self.grid.shape[1] - 1, self.pos_col + iCol))
        b = (self.pos_row, self.pos_col)
        if not a == b:
            self.last_move = True
        else:
            self.last_move = False
        return b

    def is_out_borders(self, movement, pos):
        out_of_border=False
        if pos == 'row':
            if movement in [1,2,3]:
                if movement==1 and (self.pos_row==0 or self.pos_col==0):
                    out_of_border = True
                if movement==3 and (self.pos_row==0 or self.pos_col==self.n_col-1):
                    out_of_border = True
                if movement==2 and self.pos_row==0:
                    out_of_border= True
            elif movement in [7,8,9] :
                if movement==9 and (self.pos_row==self.n_row-1 or self.pos_col==self.n_col-1):
                    out_of_border = True
                if movement==7 and (self.pos_row==self.n_row-1 or self.pos_col==0):
                    out_of_border = True
                if movement==8 and self.pos_row==self.n_row-1:
                    out_of_border= True
            else:
                out_of_border = False
        elif pos == 'col':
            if movement in [1,4,7]:
                if movement==1 and (self.pos_row==0 or self.pos_col==0):
                    out_of_border = True
                if movement==7 and (self.pos_row==self.n_row-1 or self.pos_col==0):
                    out_of_border = True
                if movement==4 and self.pos_col==0:
                    out_of_border= True
            elif movement in [3,6,9]:
                if movement==3 and (self.pos_row==0 or self.pos_col==self.n_col-1):
                    out_of_border = True
                if movement==9 and (self.pos_row==self.n_row-1 or self.pos_col==self.n_col-1):
                    out_of_border = True
                if movement==6 and self.pos_col==self.n_col-1:
                    out_of_border= True
            else:
                out_of_border = False
        
        return out_of_border      
        
    def __render__(
                    self, 
                    title='Forest Fire Automaton',
                    show=True, 
                    wait_time=-1): #RWH. Little additions to the render method
        # Before named render_frame()
            # Plot style
            sns.set_style('whitegrid')
            # Main Plot
            if show and wait_time > 0:
                plt.ion()
            fig = plt.imshow(self.grid_to_rgba(), aspect='equal', animated=False)
            # Title showing Reward by default
            if title ==  '':
                plt.title('Reward {}'.format(np.round(self.calculate_reward(), 2)))
            else:
                # Title showing Reward
                plt.title(title)#, **self.title_font)
            # Modify Axes
            ax = plt.gca()
            # Major ticks
            #ax.set_xticks(np.arange(0, self.n_col, 1))
            #ax.set_yticks(np.arange(0, self.n_row, 1))
            # Labels for major ticks
            #ax.set_xticklabels(np.arange(0, self.n_col, 1))#, **self.axes_font)
            #ax.set_yticklabels(np.arange(0, self.n_row, 1))#, **self.axes_font)
            # Minor ticks
            #ax.set_xticks(np.arange(-.5, self.n_col, 1), minor=True)
            #ax.set_yticks(np.arange(-.5, self.n_row, 1), minor=True)
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
            ax.grid(which='major', color='w', linestyle='-', linewidth=0)
            ax.tick_params(axis=u'both', which=u'both',length=0)
            # Add Helicopter Cross
            marker_style = dict(color='0.7', marker='P',
                        markersize=12, markerfacecolor='0.2')
            ax.plot(self.pos_col, self.pos_row, **marker_style)
            fig = plt.gcf()
            if show:
                if wait_time > 0:
                    plt.draw()
                    plt.pause(wait_time)
                    plt.close('all')
                else:
                    plt.show()
            return fig

    #@gif.frame
    def _get_image(self, title=''): #RWH
        # Saves a frame on the buffer of frames of the object
        fig = self.__render__(show=False, title=title)
        fig.canvas.draw()      # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #self.frames.append(image)
        plt.clf()
        return image
    
class EnvMakerForestFire(Helicopter, Env):
    """
    Implementation of a class to generate multiple Environments
    for a Reinforcement Learning task.

    All the created environments follow the
    Open AI gym API:

    env = EnvMakerForestFire()

    env.reset()
    env.step(action)
    env.render()
    env.close()

    The created environment simulates a helicopter trying to extinguish a forest fire.
    The forest is simulated using a Forest Fire Automaton [Drossel and Schwabl (1992)] and
    the helicopter as a position on top of the lattice and some effect over the cells.
    At each time step the Agent has to make a decision to where in the neighborhood to move the helicopter,
    then the helicopter moves and has some influence over the destination cell,
    the effect is simply changing it to another cell type, usually from 'fire' to 'empty'
    and the reward is some function of the current state of the system,
    usually just counting cells types, multiplying for some weights (positive for trees and negative for fires) and adding up.

    The actions to move the helicopter are the natural numbers from 1 to 9, each representing a direction:
        1. Left-Up
        2. Up
        3. Right-Up
        4. Right
        5. Don't move
        6. Left
        7. Left-Down
        8. Down
        9. Right-Down

    Forest Fire Automaton Drossel and Schwabl (1992)

    Three type of cells: TREE, EMPTY and FIRE
    At each time step and for each cell apply the following rules
    (order does not matter).
        * With probability f:                       Lighting Rule
            TREE turns into Fire
        * If at least one neighbor is FIRE:         Propagation Rule
            TREE turns into Fire
        * Unconditional:                            Burning Rule
            FIRE turns into EMPTY
        * With probability p:
            EMPTY turns into TREE                    Growth Rule

    Also two more cells were added.
    ROCK, does not interacts with anything
        Used as a true death cell
        Used on the Deterministic mode
        Used on the invariant boundary conditions
    LAKE, does not interacts with anything
        Used on other classes that inherit from ForestFire

    Deterministic mode: The automaton does not computes
    the Lighting and Growth rules, stops when there are
    no more FIRE cells.

    The observations gotten from the environment are a tuple of:

        1. 2D np-array with the current state of the cellular automaton grid.
        2. np-array with the position of the helicopter [row, col].
        3. np-array with the remaining moves of the helicopter until the next automaton update [moves].

    Parameters
    ----------

    env_mode : {'stochastic', 'deterinistic'}, default='stochastic'
        Main mode of the agent.
        - 'stochastic'
        Applies all the rules of the Forest Fire Automaton and sets the optional parameters (except if manually changed):
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'extinguish', moves_before_updating = ceil((n_row + n_col) / 4)
                 termination_type = 'continuing', ip_tree = 0.75, ip_empty = 0.25, ip_fire = 0.0, ip_rock = 0.0, ip_lake = 0.0
        - 'deterministic'
        Does not apply the stochastic rules of the Fire Forest Automaton, those are the Lighting and Growth rules.
        Also sets the following parameters:
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'clearing', moves_before_updating = 0
        termination_type = 'no_fire', ip_tree = 0.59, ip_empty = 0.0, ip_fire = 0.01, ip_rock = 0.40, ip_lake = 0.0
    n_row : int, default=16
        Rows of the grid.
    n_col : int, default=16
        Columns of the grid.
    p_tree : float, default=0.300
        'p' probability of a new tree.
    p_fire : float, default=0.003
        'f' probability of a tree catching fire.
    custom_grid : numpy like matrix, defult=None
        If matrix provided, it would be used as the starting lattice for the automaton
        instead of the randomly generated one. Must use the same symbols for the cells.
    pos_row : int, optional
        Row position of the helicopter.
    pos_col : int, optional
        Column position of the helicopter.
    moves_before_updating : int, optional
        Steps the Agent can make before an Automaton actuliazation.
    termination_type : {'continuing', 'no_fire', 'steps', 'threshold'}, optional
        Termination conditions for the task.
        - 'continuing'
        A never ending task.
        - 'no_fire'
        Teminate whenever there are no more fire cells, this
        is the only value that works well with env_mode='deterministic'.
        - 'steps'
        Terminate after a fixed number of steps have been made.
        - 'threshold'
        Terminate after a fixed number of cells have been or are fire cells.
    steps_to_termination: Steps to termination, optional, default=128
        Only valid with termination_type='steps'.
    fire_threshold : Accumulated fire cell threshold, optional, default=1024
        Only valid with termination_type='threshold'.
    reward_type : {'cells', 'hits', 'both', 'duration'}, optional, defualt='cells'
        Specifies the general behavior of the reward function.
        - 'cells'
        Reward that depends only in the lattice state of the Automaton.
        Multiplies the count of each cell type for a weight and then adds up all the results.
        - 'hits'
        Each time the Agent moves to a fire cell it gets rewarded.
        - 'both'
        Combines the two previous schemes.
        - 'duration'
        Returns the current step number as a reward, only useful for termination_type='threshold'.
    reward_tree : float, default=1.0
        How much each invidual tree cell gets rewarded.
    reward_fire : float, default=-10.0
        How much each individual fire cell gets rewarded.
    reward_empty : float, defualt=0.0
        How much each individual empty cell gets rewarded.
    reward_hit : float, defualt=10.0
        Reward when moving to a fire cell.
    tree : object, default=0
        Symbol to represent the tree cells.
    empty : object, default=1
        Symbol to represent the empty cells.
    fire : object, default=2
        Symbol to represent the fire cells.
    rock : object, default=3
        Symbol to represent the rock cells.
    lake : object, default=4
        Symbol to represent the lake cells.
    observation_mode : {'plain', 'one_hot', 'channels', 'channels3', 'channels4'}, default='one_hot'
        How to return the grid observation.
        - plain
        The step method returns the observation grid as a matrix of the the cells symbols.
        - one_hot
        The step method returns the observation grid as a matrix
        with entries of the cells symbols on one hot encoding. In the following way:
            tree: [1,0,0,0,0]
            empty: [0,1,0,0,0]
            fire: [0,0,1,0,0]
            rock: [0,0,0,1,0]
            lake: [0,0,0,0,1]
        - channels
        The step method returns the observation grid as a ndarray of 5 channels (5 matrices).
        A channel per cell type (5).
        On each channel, `1` marks the prescence of that cell type at that location and `0` otherwise.
        - channels3
        Same as `'channels'`, but only returns the first three channels.
        Useful when the environment will only yield tree, empty or fire cells.
        - channels4
        Same as `'channels'`, but only returns the first four channels.
        Useful when the environment will only yield tree, empty, fire or rock cells.
    sub_tree : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is 'empty'.
    sub_empty : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_fire : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is 'empty'.
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_rock : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_lake : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    ip_tree : float, optional
       Initialization probability of a tree cell.
    ip_empty : float, optional
       Initialization probability of an empty cell.
    ip_fire : float, optional
       Initialization probability of a fire cell.
       When in env_mode='deterministic', at least 1 fire cell is forced onto the grid.
    ip_rock : float, optional
       Initialization probability of a rock cell.
    ip_lake : float, optional
       Initialization probability of a lake cell.

    Methods
    ----------
    EnvMakerForestFire.reset()
        Initializes the environment and returns the first observation
        Input :
        Returns :
        tuple (grid, position)
            - grid
            np array with Automaton lattice
            - position
            np array with (row, col)

    EnvMakerForestFire.step(action) :
        Computes a step in the system
        Input :
            - action, int {1,2,3,4,5,6,7,8,9}
        Returns :
        {observation, reward, termination, information}
            - observation
            tuple with observations from the environment (grid, position)
            - reward
            float with the reward for the actions
            - termination
            bool True is the task has already ended, False otherwise
            - infomation
            dict with extra data about the system

    EnvMakerForestFire.render()
        Human friendly visualization of the system
        Input :
        Returns :
        matplotlib object

    EnvMakerForestFire.close()
        Closes the environment, prints a message
        Input:
        Returns:

    Examples
    --------

    Instantiation
    >>> env = EnvMakerForestFire()

    Starts the environment and gets first observation
    >>> grid, position = env.reset()

    Visualization
    >>> env.render()

    Performs 1 random step over the environment and assigns the results
    >>> import numpy as np
    >>> actions = list(env.movement_actions)
    >>> action = self.rg.choice(actions)
    >>> obs, reward, terminated, info = env.step(action)
    >>> env.render()

    Closes the environment
    >>> env.close
"""
    # Metadata
    version = 'v2.5.0' #RWH new version branch
    # Global Info
    terminated = False
    first_termination = True
    total_hits = 0
    total_burned = 0
    total_reward = 0.0
    steps = 0   
    # Defaults
    def_steps_to_termination = 128
    def_fire_threshold = 1024
    # Default weights set to a minimization of the reward function. RWH
    def __init__(self, env_mode = 'stochastic',
                 n_row = 16, n_col = 16, p_tree = 0.0100, p_fire = 0.003, custom_grid = None,
                 init_pos_row = None, init_pos_col = None, moves_before_updating = None,
                 termination_type = "steps", steps_to_termination = None, fire_threshold = None,
                 reward_type = 'hit', reward_tree = 1.0, reward_fire = -0.05, reward_empty = -0.0, 
                 reward_hit = 0.1, reward_step = 0.0, reward_move = 0.0, tree = 1, empty = 0, fire = 2, rock = 3, lake = 4, 
                 observation_mode = 'followGridImg', observation_shape = (84,84), sub_tree = None, sub_empty = None, sub_fire = None, 
                 sub_rock = None, sub_lake = None, ip_tree = None, ip_empty = None, ip_fire = None, 
                 ip_rock = None, ip_lake = None,pos_row=None,pos_col=None,
                 is_copy=False):

        kw_params = {
            'init_pos_row': init_pos_row, 'init_pos_col': init_pos_col,
            'n_row': n_row, 'n_col': n_col, 'p_tree': p_tree, 'p_fire': p_fire,
            'forest_mode': env_mode, 'custom_grid': custom_grid,
            'tree': tree, 'empty': empty, 'fire': fire, 'rock': rock, 'lake': lake,
            'ip_tree': ip_tree, 'ip_empty': ip_empty, 'ip_fire': ip_fire, 'ip_rock': ip_rock, 'ip_lake': ip_lake,
            'is_copy':is_copy}

        # Helicopter Initialization
        Helicopter.__init__(self, **kw_params)

        self.env_mode = env_mode

        # Effect over cells, substitution rules
        self.sub_tree, self.sub_empty, self.sub_fire, self.sub_rock, self.sub_lake = sub_tree, sub_empty, sub_fire, sub_rock, sub_lake

        # Automatic initialization according to env_mode. It sets: moves_before_updating, termination_type and default substitutions
        self.init_env_mode(moves_before_updating, termination_type)

        # Initialization of the cell substitution rules dictionary, for the effects of the helicopter
        if not is_copy:
            self.init_effects_dict()
        else:
            self.effects_dict = None

        # Counter for updating the grid
        self.remaining_moves = self.moves_before_updating

        # Termination type specific params
        self.steps_to_termination = steps_to_termination
        self.fire_threshold = fire_threshold

        # Reward params
        self.reward_type = reward_type
        self.reward_tree = reward_tree
        self.reward_fire = reward_fire
        self.reward_empty = reward_empty
        self.reward_hit = reward_hit
        self.reward_step = reward_step
        self.reward_move = reward_move

        # Grid Observations and One Hot Encoding
        self.observation_mode = observation_mode
        self.obsShape = observation_shape
        self.onehot_translation = {self.tree: [1,0,0,0,0],
                      self.empty: [0,1,0,0,0],
                      self.fire: [0,0,1,0,0],
                      self.rock: [0,0,0,1,0],
                      self.lake: [0,0,0,0,1]}

        self.imgValues = {self.tree: TREE,
                      self.empty: EMPTY,
                      self.fire: FIRE,
                      self.rock: 0,
                      self.lake: 0}
        
        # Checkpoints of the enviroment
        self.checkpoints = []
        self.checkpoint_counter = 0

    def init_env_mode(self, moves_before_updating, termination_type):
        if self.env_mode == 'stochastic':
            speed = math.ceil((self.n_row + self.n_col) / 4)
            self.sub_fire = 'empty' if self.sub_fire is None else self.sub_fire
            self.moves_before_updating = speed if moves_before_updating is None else moves_before_updating
            self.termination_type = 'continuing' if termination_type is None else termination_type

        elif self.env_mode == 'deterministic':
            self.sub_tree = 'empty' if self.sub_fire is None else self.sub_fire
            self.moves_before_updating = 0 if moves_before_updating is None else moves_before_updating
            self.termination_type = 'no_fire' if termination_type is None else termination_type
        else:
            raise ValueError('Unrecognized Environment Mode')

    def init_effects_dict(self):
        effect_translation={
            'tree': self.tree,
            'empty': self.empty,
            'fire': self.fire,
            'rock': self.rock,
            'lake': self.lake}
        effect_over_tree = effect_translation['tree'] if self.sub_tree is None else effect_translation[self.sub_tree]
        effect_over_empty = effect_translation['empty'] if self.sub_empty is None else effect_translation[self.sub_empty]
        effect_over_fire = effect_translation['fire'] if self.sub_fire is None else effect_translation[self.sub_fire]
        effect_over_rock = effect_translation['rock'] if self.sub_rock is None else effect_translation[self.sub_rock]
        effect_over_lake = effect_translation['lake'] if self.sub_lake is None else effect_translation[self.sub_lake]
        self.effects_dict={
            self.tree: effect_over_tree,
            self.empty: effect_over_empty,
            self.fire: effect_over_fire,
            self.rock: effect_over_rock,
            self.lake: effect_over_lake
            }

    def init_global_info(self):
        self.terminated = False
        self.total_hits = 0
        self.total_burned = 0
        self.total_reward = 0.0
        self.steps = 0
        try:
            delattr(self, 'reward')
        except AttributeError:
            pass

    def reset(self):
        self.init_kw_params = {
            'env_mode': self.env_mode,
            'n_row': self.n_row, 'n_col': self.n_col, 'p_tree': self.p_tree, 'p_fire': self.p_fire, 'custom_grid': self.custom_grid,
            'init_pos_row': self.init_pos_row, 'init_pos_col': self.init_pos_col, 'moves_before_updating': self.moves_before_updating,
            'termination_type': self.termination_type, 'steps_to_termination': self.steps_to_termination, 'fire_threshold': self.fire_threshold,
            'reward_type': self.reward_type, 'reward_tree': self.reward_tree, 'reward_fire': self.reward_fire, 'reward_empty': self.reward_empty, 
            'reward_hit': self.reward_hit, "observation_shape": self.obsShape,
            'tree': self.tree, 'empty': self.empty, 'fire': self.fire, 'rock': self.rock, 'lake': self.lake, 'observation_mode': self.observation_mode,
            'sub_tree': self.sub_tree, 'sub_empty': self.sub_empty, 'sub_fire': self.sub_fire, 'sub_rock': self.sub_rock, 'sub_lake': self.sub_lake,
            'ip_tree': self.ip_tree, 'ip_empty': self.ip_empty, 'ip_fire': self.ip_fire, 'ip_rock': self.ip_rock, 'ip_lake': self.ip_lake
            }

        # Rerun object method init
        self.__init__(**self.init_kw_params)
        # Restart global vars
        self.init_global_info()
        # Return observations, gym API
        return  self.observation_grid()

    def step(self, action):
        """Must return tuple with
        numpy array, int reward, bool termination, dict info
        """
        self.steps += 1

        if not self.terminated:
            # Is it time to update forest?
            if self.remaining_moves == 0:
                # Run fire simulation
                self.update()
                # Restart the counter
                self.remaining_moves = self.moves_before_updating
            else:
                self.remaining_moves -= 1

            # Move the helicopter
            self.new_pos(action)
            # Register if it has moved towards fire
            current_cell = self.grid[self.pos_row][self.pos_col]
            self.hit = True if current_cell == self.fire else False
            self.total_hits += self.hit
            # Apply the powers of the helicopter over the grid (cell substitution)
            self.effect_over_cells()
            # Calculate reward only in 'stochastic' mode
            self.reward = self.calculate_reward() if self.env_mode == 'stochastic' else 0.0

        else:
            # Calculate reward if the episode has just ended, 0.0 otherwise
            if self.first_termination:
                self.reward = self.calculate_reward()
                self.first_termination = False
            else:
                # Convert from episodic to continuing task by always returning 0.0 reward if the episode is over
                self.reward = 0.0

        # Check for stopping condition
        self.terminated = self.is_task_terminated()

        # Update some global info
        self.total_reward += self.reward
        self.total_burned += self.count_cells()[self.fire]

        # Observations for gym API
        #obs_grid = self.observation_grid()
        #self.obs = (obs_grid, np.array([self.pos_row, self.pos_col]), np.array([self.remaining_moves]))
        self.obs = self.observation_grid()
        # Info for gym API
        info = {'steps': self.steps, 'total_reward': self.total_reward,
                'total_hits': self.total_hits, 'total_burned': self.total_burned}
        # Gym API
        return (self.obs, self.reward, self.terminated, info)

    def close(self):
        print('Gracefully Exiting, come back soon')
        self.__del__
        return None

    def __del__(self):
        return True

    def is_task_terminated(self):
        if self.termination_type == 'continuing':
            terminated = False
        elif self.termination_type == 'no_fire':
            self.count_cells()
            terminated = False if self.cell_counts[self.fire] != 0 else True
        elif self.termination_type == 'steps':
            if self.steps_to_termination is None: self.steps_to_termination = self.def_steps_to_termination
            terminated = False if self.steps < self.steps_to_termination else True
        elif self.termination_type == 'threshold':
            if self.fire_threshold is None: self.fire_threshold = self.def_fire_threshold
            terminated = False if self.total_burned < self.fire_threshold else True
        else:
            raise ValueError('Unrecognized termination parameter')
        return terminated

    def effect_over_cells(self):
        row = self.pos_row
        col = self.pos_col
        current_cell = self.grid[row][col]
        # Make the substituion of current cell, following effects_dict
        for symbol in self.effects_dict:
            if symbol == current_cell:
                self.grid[row][col] = self.effects_dict[symbol]
                break

    def calculate_reward(self):
        reward = 0.0
        self.count_cells()
        if self.reward_type == 'cells':
            reward += self.cell_counts[self.tree] * self.reward_tree
            reward += self.cell_counts[self.empty] * self.reward_empty
            reward += self.cell_counts[self.fire] * self.reward_fire
        elif self.reward_type == 'hits':
            reward += self.hit * self.reward_hit
        elif self.reward_type == 'both':
            reward += self.cell_counts[self.tree] * self.reward_tree
            reward += self.cell_counts[self.empty] * self.reward_empty
            reward += self.cell_counts[self.fire] * self.reward_fire
            reward += self.hit * self.reward_hit
        elif self.reward_type == 'duration':
            reward += self.steps
        elif self.reward_type == 'custom': #RWH
            # Custom COST function.
            reward += self.cell_counts[self.tree] * self.reward_tree
            reward += self.cell_counts[self.fire]**2 * self.reward_fire
            reward += self.hit * self.reward_hit
            reward += self.cell_counts[self.empty] * self.reward_empty
            reward += self.reward_move if self.last_move else 0
        elif self.reward_type == 'quad':
            diff = self.cell_counts[self.tree] - self.cell_counts[self.fire]
            reward += math.copysign(diff**2, diff) * self.reward_tree
            #reward += self.cell_counts[self.fire] * self.reward_fire
            reward += (self.hit*2.0 - self.last_move*1.0) * self.reward_hit
            reward += self.cell_counts[self.empty] * self.reward_empty
        elif self.reward_type == 'ratioLD':
            ratio = 0.0
            if self.cell_counts[self.fire] > 0:
                ratio = self.cell_counts[self.tree] / self.cell_counts[self.fire]
            else:
                ratio = self.n_row*self.n_col
            #reward += self.reward_tree*math.exp(-ratio)
            reward += self.reward_tree*(ratio)
            reward += (self.hit*2.0 - self.last_move*1.0) * self.reward_hit
            reward += self.cell_counts[self.empty] * self.reward_empty
        elif self.reward_type == "ratio":#new
            tot = self.grid.shape[0] * self.grid.shape[1]
            reward = (self.cell_counts[self.tree] - self.cell_counts[self.fire]) / tot #v0
            #tot = (self.grid.shape[0] * self.grid.shape[1])
            #reward = (self.cell_counts[self.tree]) / tot # v1
        elif self.reward_type == "hit":
            tot = (self.grid.shape[0] * self.grid.shape[1]) * 2 # w Ratio
            reward = (self.cell_counts[self.tree]) / tot
            reward += self.hit * 0.5  
        else:
            raise ValueError('Unrecognized reward type')
        return reward

    def count_cells(self):
        cell_types, counts = np.unique(self.grid, return_counts=True)
        cell_counts = defaultdict(int, zip(cell_types, counts))
        self.cell_counts = cell_counts
        return cell_counts

    def random_policy(self):
        actions = list(self.movement_actions)
        action = self.rg.choice(actions)
        return action

    def observation_grid(self):
        if self.observation_mode == 'plain':
            return self.grid
        elif self.observation_mode == 'one_hot':
            return self.get_onehot_forest()
        elif self.observation_mode == 'channels':
            return self.get_channels_forest()
        elif self.observation_mode == 'channels3':
            return self.get_channels_forest()[:3]
        elif self.observation_mode == 'channels4':
            return self.get_channels_forest()[:4]
        elif self.observation_mode == "followGridImg":
            return self.followGridObs(True)
        elif self.observation_mode == "staticGridImg":
            return self.staticGridObs(True)
        elif self.observation_mode == "followGridPos":
            return self.followGridObs(False)
        elif self.observation_mode == "staticGridPos":
            return self.staticGridObs(False)
        else:
            raise ValueError("Bad Observation Mode.\nTry: {'plain', 'one_hot', 'channels'}")
    
    @staticmethod
    @numba.njit
    def quickGrid(grid:np.ndarray, fire, tree, empty, rock, lake):
        """
        Hard coded values for the cell type in gridImg
        """
        img = np.zeros(grid.shape, dtype=np.uint8)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                currentCell, val = grid[row,col], 0
                if currentCell == fire:
                    val = FIRE
                elif currentCell == tree:
                    val = TREE
                else:
                    val = EMPTY
                img[row, col] = val
        return img
    
    def getImgGrid(self, display_agent = True):
        """Returns a ndarray uint8 with a imagen representation 
        of all the grid"""
        gridShape = self.grid.shape[:2]
 
        rowBig = gridShape[0] > self.obsShape[0]
        colBig = gridShape[1] > self.obsShape[1]
        
        img = self.quickGrid(self.grid, 
                                self.fire, self.tree, self.empty, self.rock, self.lake)
        if display_agent:
            img[self.pos_row, self.pos_col] = 255 # Agent Helicopter is the brigthest spot on the plane

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

    def followGridObs(self, display_agent):
        """
        Follows the agent through the grid
        """
        gridShape = self.grid.shape[:2]
        
        img = self.getImgGrid(display_agent)
        
        if 0 > (self.pos_row - math.floor(self.obsShape[0] / 2)):
            rMin, rMax = 0, self.obsShape[0]
        elif (self.pos_row + math.ceil(self.obsShape[0] / 2)) >= gridShape[0]:
            rMax = gridShape[0]
            rMin = rMax - self.obsShape[0]
        else:
            rMin = self.pos_row - math.floor(self.obsShape[0] / 2)
            rMax = rMin + self.obsShape[0]

        if 0 > (self.pos_col - math.floor(self.obsShape[1] / 2)):
            cMin, cMax = 0, self.obsShape[1]
        elif (self.pos_col + math.ceil(self.obsShape[1] / 2)) >= gridShape[1]:
            cMax = gridShape[1]
            cMin = cMax - self.obsShape[1]
        else:
            cMin = self.pos_col - math.floor(self.obsShape[1] / 2)
            cMax = cMin + self.obsShape[1]

        img = img[rMin:rMax, cMin:cMax]

        if display_agent:
            return img
        return {"frame":img, "position":(self.pos_row, self.pos_col)}

    def staticGridObs(self, display_agent):
        img = self.getImgGrid(display_agent)
        gridShape = self.grid.shape[:2]
        #sectorR = math.ceil(gridShape[0] / (self.obsShape[0] - 2))
        #sectorC = math.ceil(gridShape[1] / (self.obsShape[1] - 2))
        agentRSec = self.pos_row // (self.obsShape[0] - 2)
        agentCSec = self.pos_col // (self.obsShape[1] - 2)
        rMax = min(gridShape[0], (self.obsShape[0] - 2) * (agentRSec + 1))
        cMax = min(gridShape[1], (self.obsShape[1] - 2) * (agentCSec + 1))
        rMin = max(0, rMax - self.obsShape[0] + 2)
        cMin = max(0, cMax - self.obsShape[1] + 2)

        newImg = np.zeros(self.obsShape,dtype=np.uint8)
        newImg[1:-1,1:-1] = img[rMin:rMax, cMin:cMax]
        Fire = self.imgValues[self.fire]

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

        if display_agent:
            return newImg
        return {"frame":newImg, "position":(self.pos_row, self.pos_col)}

    def _get_image____(self): # NOPE
        return self.getImgGrid()

    def render(self):
        plt.imshow(self.getImgGrid(), cmap=plt.cm.binary)

    def get_onehot_forest(self):
        onehot_grid = self.grid.tolist()
        for row in range(self.n_row):
            for col in range(self.n_col):
                current_cell = onehot_grid[row][col]
                for key in self.onehot_translation:
                    if key == current_cell:
                        onehot_grid[row][col] = self.onehot_translation[key]
                        break
        return np.array(onehot_grid)

    def get_channels_forest(self):
        grid = self.get_onehot_forest()
        return np.array([grid[:,:,channel] for channel in range(np.shape(grid)[-1])])
    
    #Features Added by Mau, & Rob
    def copy(self):
        # Does a simple copy at the actual state variables to use for 
        # sample recollection in parallel of the object.
        # It does not copy nothing related to frames.    
        # The creation of the new environment generates a new random seed.     
        grido = self.grid.copy()
        NEW = EnvMakerForestFire(
            init_pos_row=self.pos_row,init_pos_col=self.pos_col,n_row = self.n_row, 
            n_col = self.n_col,p_tree = self.p_tree, p_fire =self.p_fire,
                 moves_before_updating = self.moves_before_updating,
                 reward_type = self.reward_type, reward_tree = self.reward_tree,
                 reward_fire = self.reward_fire, reward_empty =self.reward_empty, reward_hit = self.reward_hit,
                 sub_tree = self.sub_tree, sub_empty = self.sub_empty, sub_fire = self.sub_fire, 
                 sub_rock = self.sub_rock, sub_lake = self.sub_lake,ip_tree = self.ip_tree,
                 ip_empty =self.ip_empty, ip_fire =self.ip_fire, ip_rock = self.ip_rock,
                 ip_lake = self.ip_lake, env_mode=self.env_mode, steps_to_termination=self.steps_to_termination,
                 fire_threshold=self.fire_threshold, observation_mode=self.observation_mode,
                 is_copy=True)
        NEW.grid = grido  # It needed a copy, silly me. RWH
        NEW.steps = self.steps
        NEW.total_burned = self.total_burned
        NEW.total_hits = self.total_hits
        NEW.total_reward = self.total_reward
        NEW.remaining_moves = self.remaining_moves
        NEW.first_termination = self.first_termination
        NEW.terminated = self.terminated
        NEW.last_move = self.last_move
        NEW.effects_dict = self.effects_dict.copy()
        #NEW.checkpoints = self.checkpoints.copy()
        #NEW.checkpoint_counter = self.checkpoint_counter
        return NEW
        
    def make_checkpoint(self): 
        # Function to save a state of the environment to branch
        self.checkpoints.append((
            self.grid.copy(), self.pos_row, self.pos_col, self.total_hits, self.total_reward, 
            self.remaining_moves, self.steps_to_termination, self.first_termination, self.total_burned,
            self.terminated, self.steps, self.last_move
            ))
        self.checkpoint_counter += 1
        return self.checkpoint_counter - 1
    
    def load_checkpoint(self, checkpoint_id):
        # Function to recall a previous state of the environment given the id
        if not isinstance(checkpoint_id, int):
            raise "checkpoint_id must be a valid integer number, {} was given.".format(type(checkpoint_id))
        try:
            self.grid, self.pos_row, self.pos_col, self.total_hits, self.total_reward, \
            self.remaining_moves, self.steps_to_termination, self.first_termination, self.total_burned, \
            self.terminated, self.steps, self.last_move = self.checkpoints[checkpoint_id]
            return True
        except:
            raise "checkpoint_id references to an ill checkpoint. Sorry."
            
    def available_actions(self):#, pos): RWH. This one seemed, unnecesary
        av_actions=[]
        for action in self.movement_actions:                       
            if action==1 or action==2 or action==3:                
                d=self.is_out_borders(action,pos='row')
            elif action==7 or action==8 or action==9:
                d=self.is_out_borders(action,pos='row')
            elif action==1 or action==4 or action==7:
                d=self.is_out_borders(action,pos='col')
            elif action==3 or action==6 or action==9:                
                d=self.is_out_borders(action,pos='col')
            else:
                d=False            
            if d==False:                
                av_actions.append(action)
        return av_actions
    
    def Encode(self):
        # Enconding strings for the grid just in line form
        s = str(self.pos_row) + str(self.pos_col) 
        for j in range(self.n_row):
            for i in range(self.n_col):
                cell = self.grid[j,i]
                if cell == self.tree or cell == self.empty:
                    # To make the set space a bit smaller, the two are
                    # interechanged for tree
                    cell = self.tree
                s += str(cell)
        return s

    def PaddedGrid(self):
        # Function to artificially expand the grid for Same padding
        # Perhaps needed for designing heurtistics easier.
        size = self.grid.shape
        PadGrid = np.zeros((size[0]+2,size[1]+2), dtype=np.int16)
        PadGrid[:,:] = self.empty
        PadGrid[1:-1,1:-1] = self.grid # Prop the original grid into the padded one
        return PadGrid

    @property
    def action_space(self):
        return Discrete(9)