import random

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..environments.generational_world import GenerationalWorld

WIDTH = 50
HEIGHT = 50


def create_random_escape_world(**world_kwargs):
    world = GenerationalWorld(width=WIDTH, height=HEIGHT, cost_of_life=0.2, **world_kwargs)

    agents = 0
    while agents < 50:
        nx = random.randint(0, 3)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            agent = Agent(brain, nx, ny, initial_health=100.0, color=None)
            world.add_agent(agent)
            agents += 1

    return world


def create_smart_escape_world(**world_kwargs):
    """
    Injects "scent tracker" genes
    """
    world = GenerationalWorld(width=WIDTH, height=HEIGHT, cost_of_life=0.2, **world_kwargs)

    agents = 0
    while agents < 50:
        nx = random.randint(0, 3)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            brain.W1.fill(0.0)
            brain.b1.fill(0.0)
            brain.W2.fill(0.0)
            brain.b2.fill(0.0)

            # Hidden 0: is there scent nearby?
            for i in range(9):
                if i != 4:
                    brain.W1[InputSchema.ID_SCENT + i, 0] = 1.0
            brain.b1[1] = 1.0  # wander if no scent

            # Hidden standing on food
            brain.W1[InputSchema.ID_ENERGY + 4, 2] = 10.0

            # Output: MOVE_TO_SCENT favoured
            brain.W2[0, Action.MOVE_TO_SCENT] = 100.0
            brain.W2[1, Action.MOVE_RANDOM] = 5.0

            brain.W2[2, Action.GATHER] = 110.0  # if standing on food, eats it

            agent = Agent(brain, nx, ny, initial_health=100.0, color=None)
            world.add_agent(agent)
            agents += 1

    return world
