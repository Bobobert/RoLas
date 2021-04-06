import numpy as np
from math import ceil
from numpy.linalg import norm
from math import ceil, floor, cos, pi

def init_bulldozer(grid_space, empty, tree, fire, p_tree:float = 0.6, fires:int = 1):
    assert (p_tree >= 0) and (p_tree <= 1), "p_tree must be in [0,1]"
    
    shape = grid_space.shape
    grid = np.zeros(shape, dtype=grid_space.dtype)

    for row in range(shape[0]):
        for col in range(shape[1]):
            grid[row, col] = tree if np.random.uniform() <= p_tree else empty
    
    randint = lambda x, y : np.random.randint(x,y)
    randintT = lambda x: (randint(0, x[0]), randint(0, x[1]))

    for _ in range(fires):
        firePos = randintT(shape)
        grid[firePos] = fire

    while True:
        pos = randintT(shape)
        if grid[pos] != fire:
            break

    return grid, np.array(pos)

def generate_wind_kernel(direction, speed, c1):
    """
    Function to generate the kernel for wind probabilities
    to apply into the lattice transitions.

    This version is for Moore's neighborhood only

    parameters
    ----------
    direction: int, float
        Respect to a horizontal axis, the angle in which the
        wind is going to.
    speed: int, float
        Not unit depended, as MPH or kmh. Positive amount 
        expressing the ||wind|| intensity.
    c1: int, float
        To be multiplied by the inverse. Controls how much
        the wind's speed affects its behavior to propagate
    """
    # Assertions
    assert (direction <= 360) and (direction >= 0), \
        "Sorry to bother, direction but this must be expressed in [0, 360]"
    assert (speed >= 0), \
        "Speed must be non-negative"
    assert (c1 >= 0), \
        "c1 must be non-negative"
    # Creating kernel
    MOORES = [135,90,45,180,0,0,225,270,315]
    SHAPE = (3,3)
    # Initialize every cell has the same chance to 
    # propagete its fire to
    kernel = np.ones(SHAPE, dtype = np.float32)
    cx, cy = floor(SHAPE[0] / 2), floor(SHAPE[1] / 2)
    # Adjusting with the wind vector
    deg_adjust = pi / 180
    wind_effect = speed / c1
    for i, angle in enumerate(MOORES):
        diff = abs(angle - direction) * deg_adjust
        projection = cos(diff) * wind_effect
        if projection < - 1.0:
            projection = - 1.0
        x, y = int(i // SHAPE[1]), int(i % SHAPE[1])
        kernel[x,y] += projection
    kernel[cx, cy] = 0.0 # The center of the kernel is always zero
    # Normalizing
    kernel = kernel / norm(kernel)
    # Fixing kernel
    kernel.setflags(write = False)
    # print("New Wind Kernel\n",kernel)
    return kernel