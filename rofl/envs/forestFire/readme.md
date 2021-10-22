# Forest Fire Helicopter Environment
This version is here as legacy from the orginal repo, please consult https://github.com/elbecerrasoto/gym-forest-fire for more information.

Both classes to make a forest fire environment are modified to support different type 
of observations, while using numba for some increase in perfomance using njit functions
for those nested loops. As a side effect, its quite hardcoded and is not recomended to be used
outside to reproduce some past results. 

For better environments and results, please consider using [gym_cellular_automata](https://github.com/elbecerrasoto/gym-cellular-automata).
