# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:46:08 2020

@author: ebecerra
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numba

class ForestFire():
    """
    Implementation of the Forest Fire Automaton
    using the rules proposed on Drossel and Schwabl (1992)
    
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
    
    Parameters
    ----------
    n_row : Rows of the grid, default=16
    n_col : Columns of the grid, default=16
    p_tree : 'p' probability of a new tree, default=0.100
    p_fire : 'f' probability of a tree catching fire, default=0.001
    forest_mode : Automaton mode, default='stochastic'
        - 'stochastic'
        Standard automaton
        - 'deterministic'
        Does not computes the Lighting and Growth rules, stops when there are
        no more FIRE cells.
    boundary : Boundary conditions of the automaton, default='invariant'
        - 'invariant'
        The boundaries are ROCK cells, they do nothing
        - 'reflective'
        The boundaries are a copy of the adjecent cells.
        - 'toroidal' 
        The boundaries are linked to the other end of the lattice.
    custom_grid : numpy like matrix, defult=None
        If matrix provided, it would be used as the starting lattice for the automaton
        instead of the randomly generated one. Must use the same symbols for the cells.
    tree : Representation of the TREE cell, default='|'
    empty : Representation of the EMPTY cell, default='.'
    fire : Representation of the FIRE cell, default='*'
    rock : Representation of the ROCK cell, default='#'
    lake : Representation of the LAKE cell, default='O'
    
    Methods
    ----------
    ForestFire.update() : Calculate 1 step on current grid
        - return
        The new grid
        - exception
        StopIteration when in deterministic mode and grid has no FIRE
    
    ForestFire.render() : Visualization of the grid
        - return
        The plot object
    
    ForestFire.simulate(times=10, delay=0.7) : Simulates `times` steps
    
    ForestFire.grid_init() : Initializes a new grid
    
    ForestFire.grid_init_manually(grid) : Initializes a new custom grid
    
    Attributes
    ----------
    grid : Current lattice
    init_stoch : Initialization cell probabilities for stochastic mode
        - default
        {'ip_tree':0.75, 'ip_empty':0.25, 'ip_fire':0.0, 'ip_rock':0.0, 'ip_lake':0.0}
    init_determ : Initialization cell probabilities for deterministic mode
        - default
        {'ip_tree':0.59, 'ip_empty':0.0, 'ip_fire':0.01, 'ip_rock':0.40, 'ip_lake':0.0}
    title_font : Tit;e font
        - default {'fontname':'Comfortaa'}
    axes_font : Axes font
        - default {'fontname':'Comfortaa'}
    color_tree : Green RGBA for plotting
        - defuault
        [15, 198, 43, 255])   
    color_empty : Beige RGBA for plotting
        - default
        [255, 245, 166, 255]
    color_fire : Red RGBA for plotting
        - defuault
        [255, 106, 58, 255]
    color_rock : Brown RGBA for plotting
        - defuault
        [179, 139, 109, 255]
    color_lake : Blue RGBA for plotting
        - defuault
        [131, 174, 255, 255]
        
    Examples
    --------
    >>> import forest_fire
    
    Instantiate a ForestFire object
    with probability of a new tree of 0.3 and probability of a new fire of 0.006
    
    >>> amazonia = forest_fire.ForestFire(p_tree=0.3, p_fire=0.006)
    
    Visualize the automaton
    
    >>> amazonia.render()
    
    Run 1 step
    
    >>> amazonia.update()
    >>> amazonia.render()
    
    Simulate 10 steps
    >>> amazonia.simulate(10)
    
    Instantiate a 8x12 automaton on deterministic mode
    and run it for 20 steps (probably ending the simulation)
    
    >>> amazonia = forest_fire.ForestFire(n_row=8, n_col=12, forest_mode='deterministic')
    >>> amazonia.simulate(20)
    
    Change the grid for a predifined one and run it again
    
    >>> forest_custom = [['|','#','O','*'],['.','#', '#', '|'],['|', '|','O','|']]
    >>> amazonia = forest_fire.ForestFire(custom_grid = forest_custom)    
    >>> forest.simulate(20)
    
    """
    # Render Info
    #title_font = {'fontname':'Comfortaa'}
    #axes_font = {'fontname':'Comfortaa'}
    alpha = int(1.0*255)
    color_tree = np.array([15, 198, 43, alpha]) # Green RGBA
    color_empty = np.array([255, 245, 166, alpha]) # Beige RGBA
    color_fire = np.array([255, 106, 58, alpha]) # Red RGBA
    color_rock = np.array([179, 139, 109, alpha]) # Brown RGBA
    color_lake = np.array([131, 174, 255, alpha]) # Blue RGBA
    # Grid Defualt Initialization Probabilities
    def_init_stoch = {'ip_tree':0.55, 'ip_empty':0.45, 'ip_fire':0.0, 'ip_rock':0.0, 'ip_lake':0.0}
    def_init_determ = {'ip_tree':0.59, 'ip_empty':0.0, 'ip_fire':0.01, 'ip_rock':0.40, 'ip_lake':0.0}
    def __init__(self, n_row = 16, n_col = 16, p_tree=0.100, p_fire=0.001, forest_mode = 'stochastic',
                 custom_grid=None, force_fire = True, boundary='invariant',
                 tree = 1, empty = 0, fire = 2, rock = 5, lake = 7,
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None,
                 is_copy=False): 
        
        # RWH. A random seed that changes to generate the random choices
        # Quicker method, good results, better behavior for parallel workers.
        self.rg = np.random.Generator(np.random.SFC64())

        self.n_row = n_row
        self.n_col = n_col
        self.p_tree = p_tree
        self.p_fire = p_fire
        self.forest_mode = forest_mode
        self.custom_grid = custom_grid
        self.force_fire = force_fire
        self.boundary = boundary
        # Cells
        self.tree = tree
        self.empty = empty
        self.fire =  fire
        self.rock = rock
        self.lake = lake
        # Grid random initialization
        self.ip_tree = ip_tree
        self.ip_empty = ip_empty
        self.ip_fire = ip_fire
        self.ip_rock = ip_rock
        self.ip_lake = ip_lake

        if not is_copy: #RWH
            self.grid_init()
        else:
            self.grid = None

        self.last_update = False

    def update(self):
        if self.forest_mode == 'stochastic':
            self.update_stochastic()
        elif self.forest_mode == 'deterministic':
            self.update_deterministic()
        else:
            raise Exception('Mode Invalid: Try "stochastic"|"deterministic"')
        return self.grid

    def fire_around(self, grid, row, col, boundary = 'invariant'):
        """ Checks for fire in the neighborhood of grid[row][col]"""
        if boundary == 'invariant':
            neighborhood = self.neighborhood_invariant(grid, row, col)
        elif boundary == 'reflective':
            neighborhood = self.neighborhood_reflective(grid, row, col)
        elif boundary == 'toroidal':
            neighborhood = self.neighborhood_toroidal(grid, row, col)
        else:
            raise Exception('Bad Boundary Name')
        burning_near = False
        for neighbor_state in neighborhood: 
            if neighbor_state == self.fire:
                burning_near = True
                break
        return burning_near

    
    @staticmethod
    @numba.njit
    def updateSto(grid:np.ndarray, throws:np.ndarray, pFire: float, pTree: float, 
                    fire:int, tree:int, empty:int):
        rows, cols = grid.shape
        newGrid = np.zeros(grid.shape, dtype=np.uint8)
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
    
    def update_stochastic(self, grid=None, p_tree=None, p_fire=None):
        """ Updates the grid according with Drossel and Schwabl (1992) rules"""
        throws = self.rg.uniform(size=self.grid.shape) #RWH
        self.grid = self.updateSto(np.copy(self.grid), throws, self.p_fire, self.p_tree,
                        self.fire, self.tree, self.empty)
        return self.grid

    def is_fire_over(self):
        for row in range(self.n_row):
            for col in range(self.n_col):
                if self.grid[row][col] == self.fire:
                    return False
        return True

    def update_deterministic(self):
        """Just updates using the burning rule"""
        grid = self.grid
        new_grid = grid.copy()
        for row in range(self.n_row):
            for col in range(self.n_col):
                if grid[row][col] == self.tree and self.fire_around(grid, row, col, self.boundary):
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire
                elif grid[row][col] == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty
                else:
                    # Do nothing with Rocks and Lakes
                    # Also don't use light and grow rules
                    pass
        fire_over = self.is_fire_over()
        if fire_over and self.last_update:
            print('\nFire Stoppend, End of the Simulation\n')
            raise StopIteration()
        elif fire_over:
            self.last_update = fire_over
        else:
            pass
        self.grid = new_grid
        return new_grid

    def grid_init(self):
        use_default = self.ip_tree is None and self.ip_empty is None and\
            self.ip_fire is None and self.ip_rock is None and self.ip_lake is None
        if self.custom_grid is not None:
            self.grid_init_manually(self.custom_grid)
        elif self.forest_mode == 'stochastic':
            self.grid = self.grid_random(**self.def_init_stoch) if use_default else\
                self.grid_random(self.ip_tree, self.ip_empty, self.ip_fire, self.ip_rock, self.ip_lake)
        elif self.forest_mode == 'deterministic':
            self.grid = self.grid_random(**self.def_init_determ) if use_default else\
                self.grid_random(self.ip_tree, self.ip_empty, self.ip_fire, self.ip_rock, self.ip_lake)
            # Force at least 1 fire
            if self.force_fire and np.all(self.grid != self.fire):
                self.grid_add_1_fire()
        else:
            raise ValueError('Mode Invalid: Try "stochastic"|"deterministic"')

    def grid_add_1_fire(self):
        forced_row = self.rg.choice(np.arange(self.n_row))
        forced_col = self.rg.choice(np.arange(self.n_col))
        self.grid[forced_row][forced_col] = self.fire    

    def grid_random(self, ip_tree=None, ip_empty=None, ip_fire=None, ip_rock=None, ip_lake=None):
        included_for_sharing = list()
        if ip_tree is None: ip_tree = 0.0; included_for_sharing.append(0)
        if ip_empty is None: ip_empty= 0.0; included_for_sharing.append(1)
        if ip_fire is None: ip_fire = 0.0; included_for_sharing.append(2)
        if ip_rock is None: ip_rock = 0.0; included_for_sharing.append(3)
        if ip_lake is None: ip_lake = 0.0; included_for_sharing.append(4)
        init_probs = np.array((ip_tree, ip_empty, ip_fire, ip_rock, ip_lake))
        cum_ip = init_probs.sum()
        assert np.all(np.logical_and(init_probs >= 0.0, init_probs <= 1.0)) and cum_ip <= 1.0
        remaining_mass = 1.0 - cum_ip
        non_assigned_probs = len(included_for_sharing)
        shared_mass = remaining_mass / non_assigned_probs if non_assigned_probs else 0.0
        init_probs[included_for_sharing] = shared_mass
        cells = (self.tree, self.empty, self.fire, self.rock, self.lake)
        grid = self.rg.choice(cells, size=self.n_row*self.n_col, p=init_probs).\
                    reshape(self.n_row,self.n_col)
        self.init_probabilities = dict(zip(('tree', 'empty', 'fire', 'rock', 'lake'), init_probs))
        return grid.astype(np.uint8)

    def grid_init_manually(self, grid):
        everything_good = np.all(np.logical_or.reduce((grid == self.tree,
                                                    grid == self.empty,
                                                    grid == self.fire,
                                                    grid == self.rock,
                                                    grid == self.lake)))
        if not everything_good:
            raise ValueError('Unrecognized Cell')
        self.n_row = grid.shape[0]
        self.n_col = grid.shape[1]
        self.grid = grid

    def grid_to_rgba(self):
        rgba_mat = np.copy(self.grid).tolist()
        n_row, n_col = self.grid.shape
        for row in range(n_row):
            for col in range(n_col):
                if rgba_mat[row][col] == self.tree:
                    rgba_mat[row][col] = self.color_tree
                elif rgba_mat[row][col] == self.empty:
                    rgba_mat[row][col] = self.color_empty
                elif rgba_mat[row][col] == self.fire:
                    rgba_mat[row][col] = self.color_fire
                elif rgba_mat[row][col] == self.rock:
                    rgba_mat[row][col] = self.color_rock
                elif rgba_mat[row][col] == self.lake:
                    rgba_mat[row][col] = self.color_lake
                else:
                    ax = rgba_mat[row,col]
                    print(ax, type(ax), self.tree, type(self.tree))
                    raise ValueError('Error: Unidentified cell')
        rgba_mat = np.array(rgba_mat)
        return rgba_mat

    def is_bound_legal(self, row, col, grid=None):
        """Check borders of a target cell"""
        if grid is None:
            grid = self.grid
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        r_offset = row + np.array([-1, 1])
        c_offset = col + np.array([-1, 1])
        up = r_offset[0] >= 0
        down = r_offset[1] <= n_row-1
        left = c_offset[0] >= 0
        right = c_offset[1] <= n_col-1
        return {'up': up, 'down': down, 'left': left, 'right': right}
        
    def neighborhood_invariant(self, grid, row, col):
        """Invariant Boundary Conditions"""
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        legal_bounds = self.is_bound_legal(row, col)
        up_left = [grid[(row-1),(col-1)]]\
            if legal_bounds['up'] and legal_bounds['left'] else [self.rock]
        up_center = [grid[(row-1),col]]\
            if legal_bounds['up'] else [self.rock]
        up_right = [grid[(row-1)%n_row,(col+1)%n_col]]\
            if legal_bounds['up'] and legal_bounds['right'] else [self.rock]
        middle_left = [grid[row,(col-1)%n_col]]\
            if legal_bounds['left'] else [self.rock]
        middle_right = [grid[row,(col+1)%n_col]]\
            if legal_bounds['right'] else [self.rock]
        down_left = [grid[(row+1)%n_row,(col-1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['left'] else [self.rock]
        down_center = [grid[(row+1)%n_row,col]]\
            if legal_bounds['down'] else [self.rock]
        down_right = [grid[(row+1)%n_row,(col+1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['right'] else [self.rock]
        neighborhood = up_left + up_center + up_right +\
            middle_left + middle_right +\
            down_left + down_center + down_right
        return neighborhood       
    def neighborhood_reflective(self, grid, row, col):
        """Reflective Boundary Conditions"""
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        legal_bounds = self.is_bound_legal(row, col)
        up_left = [grid[(row-1),(col-1)]]\
            if legal_bounds['up'] and legal_bounds['left']\
            else [grid[(row-1),(col)]] if legal_bounds['up']\
            else [grid[(row),(col-1)]] if legal_bounds['left']\
            else [grid[(row),(col)]] 
        up_center = [grid[(row-1),col]]\
            if legal_bounds['up']\
            else [grid[(row),col]]
        up_right = [grid[(row-1)%n_row,(col+1)%n_col]]\
            if legal_bounds['up'] and legal_bounds['right']\
            else [grid[(row-1),(col)]] if legal_bounds['up']\
            else [grid[(row),(col+1)]] if legal_bounds['right']\
            else [grid[(row),(col)]]
        middle_left = [grid[row,(col-1)%n_col]]\
            if legal_bounds['left']\
            else [grid[(row),col]]
        middle_right = [grid[row,(col+1)%n_col]]\
            if legal_bounds['right']\
            else [grid[(row),col]]
        down_left = [grid[(row+1)%n_row,(col-1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['left']\
            else [grid[(row+1),(col)]] if legal_bounds['down']\
            else [grid[(row),(col-1)]] if legal_bounds['left']\
            else [grid[(row),(col)]]
        down_center = [grid[(row+1)%n_row,col]]\
            if legal_bounds['down']\
            else [grid[(row),col]]
        down_right = [grid[(row+1)%n_row,(col+1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['right']\
            else [grid[(row+1),(col)]] if legal_bounds['down']\
            else [grid[(row),(col+1)]] if legal_bounds['right']\
            else [grid[(row),(col)]]
        neighborhood = up_left + up_center + up_right +\
            middle_left + middle_right +\
            down_left + down_center + down_right 
        return neighborhood       
    def neighborhood_toroidal(self, grid, row, col):
        """Periodic Boundary Conditions (toroidal)"""
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        up_left = [grid[(row-1)%n_row, (col-1)%n_col]]
        up_center = [grid[(row-1)%n_row, col]]
        up_right = [grid[(row-1)%n_row, (col+1)%n_col]]
        middle_left = [grid[row, (col-1)%n_col]]
        middle_right = [grid[row, (col+1)%n_col]]
        down_left = [grid[(row+1)%n_row, (col-1)%n_col]]
        down_center = [grid[(row+1)%n_row, col]]
        down_right = [grid[(row+1)%n_row, (col+1)%n_col]]
        neighborhood = up_left + up_center + up_right +\
            middle_left + middle_right +\
            down_left + down_center + down_right
        return neighborhood

    def observation_grid(self):
        raise NotImplementedError

    def seed(self, s):
        self.rg = np.random.Generator(np.random.SFC64(s))
        return [s]
