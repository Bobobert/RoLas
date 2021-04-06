import numpy as np

from gym import spaces
from rofl.envs.gym_cellular_automata import Operator

# ------------ Forest Fire Coordinator


class ForestFireCoordinator(Operator):
    is_composition = True
    last_lattice_update = 0

    def __init__(
        self,
        cellular_automaton,
        modifier,
        mdp_time,
        grid_space=None,
        action_space=None,
        context_space=None,
    ):

        self.suboperators = cellular_automaton, modifier
        self.cellular_automaton, self.modifier = cellular_automaton, modifier

        self.times = mdp_time

        if grid_space is None:
            assert (
                cellular_automaton.grid_space is not None
            ), "grid_space could not be inferred"

            self.grid_space = cellular_automaton.grid_space

        if action_space is None:
            assert (
                modifier.action_space is not None
            ), "action_space could not be inferred"

            self.action_space = modifier.action_space

        self.context_space = context_space

    def update(self, grid, action, context):
        # Bulldozers do its modifications first, as it starts consuming 
        # time with the lattice, this can change just after the 
        grid, context = self.modifier(grid, action, context)

        # Time after bulldozer
        _, internal_time, _ = context
        lattice_time = self.times["lattice"]
        if internal_time >= (self.last_lattice_update + lattice_time):
            # Do the respective updates
            # Lattice updates are strict times "lattice_time"
            for _ in range((internal_time - self.last_lattice_update) // lattice_time):
                grid, _ = self.cellular_automaton(grid, action, context)
                # Update context from the given lattice update. 
                # If alive becomes true this will not change again
                grid, context = self.modifier.bulldozer_status(grid, context)
            # From the due updates, sync the time
            self.last_lattice_update = internal_time - internal_time % lattice_time

        return grid, context
