"""
    EnvMakers are functions that return a function which creates
    and environment in a set of environments. These should return
    the (environment, [seed:int]) 
"""

from .gym import gymEnvMaker, atariEnvMaker, gymcaEnvMaker
from .CA import forestFireEnvMaker
