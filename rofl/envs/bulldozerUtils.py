from rofl.envs.forestFire.helicopter import EMPTY
from numpy import empty
from rofl.functions.const import *
import numba

def initWindKernel(direction, speed, c1):
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
    norm = np.linalg.norm
    # Creating kernel
    MOORES = [135,90,45,180,0,0,225,270,315]
    SHAPE = (3,3)
    # Initialize every cell has the same chance to 
    # propagete its fire to
    kernel = np.ones(SHAPE, dtype = np.float32)
    cx, cy = floor(SHAPE[0] / 2), floor(SHAPE[1] / 2)
    # Adjusting with the wind vector
    deg_adjust = PI / 180
    wind_effect = speed / c1
    for i, angle in enumerate(MOORES):
        diff = abs(angle - direction) * deg_adjust
        projection = math.cos(diff) * wind_effect
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

TREE, EMPTY, FIRE, BURNT, AGENT = 100, 0, 200, 50, 255

@numba.njit
def quickGrid(grid:ARRAY, shape,
                fire, tree, empty, burnt,
                lRow, hRow, lCol, hCol,
                posRow, posCol, displayAgent:bool,
                offsetRow, offsetCol) -> ARRAY:
    """
    Hard coded values for the cell type in gridImg

    return
    ------
    img: np.ndarray
    """
    img = np.zeros(shape, dtype=np.uint8)
    i = offsetRow
    for row in range(lRow, hRow):
        j = offsetCol
        for col in range(lCol, hCol):
            currentCell, val = grid[row,col], 0
            if currentCell == fire:
                val = FIRE
            elif currentCell == tree:
                val = TREE
            elif currentCell == burnt:
                val = BURNT
            elif (row == posRow) and (col == posCol) and displayAgent:
                val = AGENT
            else:
                val = EMPTY
            img[i, j] = val
            j += 1
        i += 1
    return img

@numba.njit
def quickGridC(grid:ARRAY, shape,
                fire, tree, empty, burnt,
                lRow, hRow, lCol, hCol,
                posRow, posCol,
                offsetRow, offsetCol) -> ARRAY:
    """
    Hard coded values for the cell type in gridImg

    return
    ------
    img: np.ndarray with 4 channels
        One for fire, tree, burnt, agent
    """
    img = np.zeros((shape[0], shape[1], 4), dtype=np.bool_)
    i = offsetRow
    for row in range(lRow, hRow):
        j = offsetCol
        for col in range(lCol, hCol):
            currentCell = grid[row,col]
            if currentCell == fire:
                img[i, j, 0] = 1
            elif currentCell == tree:
                img[i, j, 1] = 1
            elif currentCell == burnt:
                img[i, j, 2] = 1
            if (row == posRow) and (col == posCol):
                img[i, j, 3] = 1
            j += 1
        i += 1
    return img

def checkSize(grid, obsShape) -> tuple[tuple, bool, bool]:
    gridShape = grid.shape[:2]
    rowBig = gridShape[0] > obsShape[0]
    colBig = gridShape[1] > obsShape[1]
    return gridShape, rowBig, colBig

def grid2ImgFollow(env, grid, context, obsShape, 
                channels: bool = False,
                displayAgent: bool = True) -> ARRAY:
    """
    Makes a window of size obsShape to display the agent in two
    ways, img and channels. The first is a grey scale image np.uint8,
    and the latter a np.bool_ type with 4 channels.

    The window always try to center the agent. if the original image
    has a smaller dimension than obsShape the image is consistently
    bigger starting to display top left.
    """
    def getInterval(pos, w, W):
        low = min(W - w, max(0, pos - ceil(w / 2)))
        return low, low + w

    img, _ = grid2Img(env, grid, context, obsShape, getInterval,
                channels, displayAgent, 0, 0)
    return img

def grid2ImgStatic(env, grid, context, obsShape, 
                channels: bool = False,
                displayAgent: bool = True) -> ARRAY:

    def getInterval(pos, w, W):
        nw = w - 2
        agentSec = pos // nw
        high = min(W, nw * agentSec + nw)
        return high - nw, high
    
    img, ((rMin, rMax), (cMin, cMax)) = grid2Img(env, grid, context, obsShape, getInterval,
                channels, displayAgent, 1, 1)
    # Change the logic, print the aid when it is not a border or edge of the screen

    rows, cols = grid.shape[:2]
    def mids(high):
        # using at least 30% of the cells
        n = ceil(high * 0.3)
        mid = ceil(high / 2)
        low = max(0, mid - ceil(n / 2))
        return [low + x for x in range(n)]

    rMids, cMids = mids(obsShape[0]), mids(obsShape[1])

    if rMin > 0:
        img[rMids, 0] = FIRE
    if rMax < (rows  - 2):
        img[rMids, -1] = FIRE
    if cMin > 0:
        img[0, cMids] = FIRE
    if cMax < (cols - 2):
        img[0, cMids] = FIRE

    return img
    

def grid2Img(env, grid, context, obsShape,
                fInterval,
                channels: bool = False,
                displayAgent: bool = True,
                offsetRow: int = 0, offsetCol: int = 0):

    params, (posRow, posCol), time = context
    (rows, cols), rowBig, colBig = checkSize(grid, obsShape)

    if rowBig:
        rMin, rMax = fInterval(posRow, obsShape[0], rows)
    else:
        rMin, rMax = 0, rows

    if colBig:
        cMin, cMax = fInterval(posCol, obsShape[1], cols)
    else:
        cMin, cMax = 0, cols

    # ca_env is not wrapped, yet
    fire, tree, empty, burnt = env._fire, env._tree, env._empty, env._burned
    lake, rock = None, None

    t = ((rMin, rMax), (cMin, cMax))

    if channels:
        return quickGridC(grid, obsShape, 
                            fire, tree, empty, burnt,
                            rMin, rMax, cMin, cMax,
                            posRow, posCol,
                            offsetRow, offsetCol), t

    return quickGrid(grid, obsShape, 
                        fire, tree, empty, burnt,
                        rMin, rMax, cMin, cMax,
                        posRow, posCol, displayAgent,
                        offsetRow, offsetCol), t
                    