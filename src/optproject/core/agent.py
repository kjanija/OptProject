import numpy as np

from .actions import Action
from .brain import Brain


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
        self.fitness = 0.0

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

        def get_strength(action_idx: Action):
            idx = int(action_idx)
            if idx >= self.brain.output_size:
                return 0.0
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


def run_agent_self_tests():
    print("#" * 50)
    print("Testing Agent class with Dynamic Coloring")

    # 1. Agent Initialization
    input_neurons = 37
    hidden_neurons = 32
    output_neurons = len(Action)

    test_brain = Brain(input_neurons, hidden_neurons, output_neurons)

    # Init WITHOUT explicit color
    agent = Agent(test_brain, x=5, y=10, initial_health=100.0, color=None)

    print(f"Created agent with genetic color: {agent.color}")
    assert len(agent.color) == 3
    assert 0.0 <= agent.color[0] <= 1.0

    # 2. Agent Reproduction
    repro_cost = 50.0
    child_agent = agent.reproduce(
        mutation_prob=0.5, mutation_amp=0.1, reproduction_cost=repro_cost
    )

    print(f"Child color: {child_agent.color}")
    # Colors should be very similar but slightly different due to mutation
    dist = np.linalg.norm(np.array(agent.color) - np.array(child_agent.color))
    print(f"Color distance (should be small): {dist}")

    print("\nAgent class tests passed successfully!")


if __name__ == "__main__":
    run_agent_self_tests()
