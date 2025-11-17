from brain import Brain

class Agent:
    """
    Represents a single agent of the A-life simulation

    It is characterised, for now, with a position, health, brain (ANN) and a
    color (representing lineages) 
    """

    def __init__(self, brain: Brain, x: int, y: int, initial_health: float, color: tuple):
        self.brain = brain
        self.x = x
        self.y = y
        self.health = initial_health
        self.color = color

    def reproduce(self, mutation_prob: float, mutation_amp: float, reproduction_cost: float) -> 'Agent':
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

        return Agent(child_brain, self.x, self.y, reproduction_cost, self.color)

    def __repr__(self):
        return f"Agent(pos=({self.x},{self.y}), health={self.health:.2f})"
    
if __name__ == "__main__":
    import numpy as np

    print("#"*50)
    print("Testing Agent class")

    # 1. Agent Initialization
    INPUT_NEURONS = 10
    HIDDEN_NEURONS = 5
    OUTPUT_NEURONS = 4
    
    test_brain = Brain(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)
    # Generate a random color
    rand_color = tuple(np.random.rand(3))
    
    agent = Agent(test_brain, x=5, y=10, initial_health=100.0, color=rand_color)
    print(f"Created agent: {agent}")
    assert agent.health == 100.0
    assert agent.x == 5
    assert agent.color == rand_color

    # 2. Agent Reproduction
    REPRO_COST = 50.0
    child_agent = agent.reproduce(mutation_prob=0.5, mutation_amp=0.1, reproduction_cost=REPRO_COST)
    
    print(f"Parent agent after reproduction: {agent}")
    assert agent.health == 100.0 - REPRO_COST
    
    print(f"New child agent created: {child_agent}")
    assert child_agent.health == REPRO_COST
    assert child_agent.color == agent.color # Same lineage
    assert child_agent.brain is not agent.brain # Different brain instance
    
    print("\nAgent class tests passed successfully!")