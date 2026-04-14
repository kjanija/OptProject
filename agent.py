from enum import IntEnum

import numpy as np

from brain import Brain


class Action(IntEnum):
    """
    List of actions available to the agent.

    If you want to modify them, modify it here, since the agent's brain will resize
    according to the number of listed actions here
    """

    GATHER = 0  # Gather resource at current pos
    GIVE = 1  # Give health to neighbour
    TAKE = 2  # Steal health to neighbour
    WAIT = 3  # Do nothing
    MOVE_TO_RESOURCE = 4  # Move to neighbouring cell with most energy
    MOVE_AWAY_FROM_AGENT = 5  # Move away from neighbouring agents
    MOVE_TOWARDS_AGENT = 6  # Move towards neighbouring agents
    MOVE_RANDOM = 7  # Random movement
    MOVE_TO_SCENT = 8


class InputSchema:
    """
    Indexes to identify the positions of various perceptions
    """

    ID_ENERGY = 0  # Energy levels in each cell in neighbouring 3x3 grid
    ID_OTHER_HEALTH = 9  # Health levels of agents in neighbouring 3x3 grid, 0 if empty
    ID_OTHER_DNA = 18  # Similarity to neighbouring agents, in (0.0, 1.0). Used to model "kinship"/"clans"
    ID_SELF_HEALTH = 27
    ID_SCENT = 28  # "scent values in 3x3 grid"

    TOTAL_INPUTS = 37


class Agent:
    """
    Represents a single agent of the A-life simulation

    It is characterised, for now, with a position, health, brain (ANN) and a
    color (representing lineages)
    """

    def __init__(
        self, brain: Brain, x: int, y: int, initial_health: float, color: tuple | None
    ):
        self.brain = brain
        self.x = int(x)
        self.y = int(y)
        self.health = initial_health
        self.age = 0
        self.reproduction_cooldown = 0

        if color is None:
            self.color = self._get_color_from_genes()
        else:
            self.color = color

    def _get_color_from_genes(self):
        """
        Maps the NNs weights to an RGB color where:
        1. Red = Agressive (Take, move towards agent)
        2. Green = Resource outputs (Gather, move to resource)
        3. Blue = Social/evasive (Give, move away)
        """
        # For this simple xAI segment we focus on the second layer

        def get_strength(action_idx: Action | None):
            w = self.brain.W2[:, action_idx]
            return np.mean(np.maximum(0, w))

        r_val = get_strength(Action.TAKE) + get_strength(Action.MOVE_TOWARDS_AGENT)
        g_val = (
            get_strength(Action.GATHER)
            + get_strength(Action.MOVE_TO_RESOURCE)
            + get_strength(Action.MOVE_TO_SCENT)
        )
        b_val = get_strength(Action.GIVE) + get_strength(Action.MOVE_AWAY_FROM_AGENT)

        tot = r_val + g_val + b_val + 1e-06
        return (r_val / tot, g_val / tot, b_val / tot)

    def reproduce(
        self, mutation_prob: float, mutation_amp: float, reproduction_cost: float
    ) -> "Agent":
        """
        Creates offspring of the agent. Offspring inherits color and [most of] the
        parents brain.
        It costs health to reproduce

        Args:
            mutation_prob (float): probability of a gene mutating
            mutation_amp (float): amplitude of the mutation
            reproduction_cost (float): health that will be removed to the parent for reproducing

        Returns:
            Brain: Brain of the offspring
        """
        self.health -= reproduction_cost
        child_brain = self.brain.clone_and_mutate(mutation_prob, mutation_amp)

        return Agent(
            child_brain, self.x, self.y, reproduction_cost, color=None
        )  # color = None in case of mutation

    def __repr__(self):
        return (
            f"Agent(pos=({self.x},{self.y}), health={self.health:.2f}, age={self.age})"
        )


if __name__ == "__main__":
    import numpy as np

    print("#" * 50)
    print("Testing Agent class with Dynamic Coloring")

    # 1. Agent Initialization
    INPUT_NEURONS = InputSchema.TOTAL_INPUTS
    HIDDEN_NEURONS = 32
    OUTPUT_NEURONS = len(Action)

    test_brain = Brain(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)

    # Init WITHOUT explicit color
    agent = Agent(test_brain, x=5, y=10, initial_health=100.0, color=None)

    print(f"Created agent with genetic color: {agent.color}")
    assert len(agent.color) == 3
    assert 0.0 <= agent.color[0] <= 1.0

    # 2. Agent Reproduction
    REPRO_COST = 50.0
    child_agent = agent.reproduce(
        mutation_prob=0.5, mutation_amp=0.1, reproduction_cost=REPRO_COST
    )

    print(f"Child color: {child_agent.color}")
    # Colors should be very similar but slightly different due to mutation
    dist = np.linalg.norm(np.array(agent.color) - np.array(child_agent.color))
    print(f"Color distance (should be small): {dist}")

    print("\nAgent class tests passed successfully!")
