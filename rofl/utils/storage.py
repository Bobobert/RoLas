"""
    Functions for agent.inRay behavior. 
    These are meant as a lazy way to not write and register
    quicker ways to serialize Tensors for mp or ray.
    Numpy buffer works much better. The tensor processed should
    be expected to not have hooks, grad, grad_fn, etc...
"""

from rofl.functions.const import TENSOR

def lazySerializer(obsDict):
    pass

def lazyDeserializer(obsDict):
    pass