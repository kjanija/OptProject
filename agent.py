from brain import Brain

class Agent:
    """
    Represents a single agent of the A-life simulation

    It is characterised, for now, with a position, health, brain (ANN) and a
    color (representing lineages) 
    """

    def __init__(self, brain: Brain, x: int, y: int, init_health: float, color: tuple):
        self.brain = brain
        self.x = x
        self.y = y
        self.health = init_health
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