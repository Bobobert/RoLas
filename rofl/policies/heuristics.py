"""
    All heuristics should be functions and have the following
    arguments to work, and to return an action as expected to 
    the policy
    
    def heuristic(environment, observation, **kwargs):
        return action
        
"""

from rofl.functions.const import *

AGENT, NORTH, SOUTH, EAST, WEST, NEAST, NWEST, SEAST, SWEST = 0, 1, 4, 3, 4, 6, 5, 8, 7

def getNeighborhoodP(env, pos_row, pos_col, radius = 3):
    rows, cols = env.n_row, env.n_col
    fire, tree, empty = env.fire, env.tree, env.empty
    """n, w, = np.zeros((radius), dtype= I_NDTYPE_DEFT), np.zeros((radius), dtype= I_NDTYPE_DEFT)
    e, s = np.zeros((radius), dtype= I_NDTYPE_DEFT),np.zeros((radius), dtype= I_NDTYPE_DEFT)
    ne, nw = np.zeros((radius, radius), dtype = I_NDTYPE_DEFT), np.zeros((radius, radius), dtype = I_NDTYPE_DEFT)
    se, sw = np.zeros((radius, radius), dtype = I_NDTYPE_DEFT), np.zeros((radius, radius), dtype = I_NDTYPE_DEFT)"""

    neighborhood = np.zeros((9,3), dtype = I_NDTYPE_DEFT)

    def add2List(typ,z):
        """
            types
            0: fire
            1: tree
            2: empty
        """
        if typ == fire:
            a = 0
        elif typ == tree:
            a = 1
        else:
            a = 2
        neighborhood[z,a] += 1

    for i in range(max(0, pos_row - radius), min(rows, pos_row + radius + 1)):
        for j in range(max(0, pos_col - radius), min(cols, pos_col + radius + 1)):
            if i == pos_row:    
                if j < pos_col:
                    z = WEST
                elif j > pos_col:
                    z = EAST
                else:
                    z = AGENT
            elif i < pos_row:
                if j < pos_col:
                    z = NWEST
                elif j > pos_col:
                    z = NEAST
                else:
                    z = NORTH
            else: # i > pos_row
                if j < pos_col:
                    z = SWEST
                elif j > pos_col:
                    z = SEAST
                else:
                    z = SOUTH
            add2List(env.grid[i,j], z)

    return neighborhood

def conservative(env, obs, radius = 3):
    pos_row, pos_col = env.pos_row, env.pos_col
    ng = getNeighborhoodP(env, pos_row, pos_col, radius)
    fires = ng[:,0]
    equals = np.equal(ng[:,0], fires.max())

    if np.sum(equals) > 1:
        # break tie in a random manner
        return nprnd.choice([i for i in range(10) if equals[i]])
    return fires.argmax()
