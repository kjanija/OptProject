import random

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..environments.generational_world import GenerationalWorld
from ..environments.twoisland_world import TwoIslandWorld
from ..environments.competitive_world import CompetitiveWorld
from ..core.config import (
    WIDTH, HEIGHT, INIT_AGENTS, INIT_HEALTH, COST_OF_LIFE,
    MIGRATORS_FRACTION, HUNTERS_HEALTH_MULTIPLIER, COST_OF_LIFE_COMPETITIVE,
    SPAWN_AREA_WIDTH
)


NUM_AGENTS = INIT_AGENTS
def create_random_escape_world(**world_kwargs):
    world = GenerationalWorld(width=WIDTH, height=HEIGHT, cost_of_life=COST_OF_LIFE, **world_kwargs)

    agents = 0
    while agents < NUM_AGENTS:
        nx = random.randint(0, SPAWN_AREA_WIDTH - 1)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            agent = Agent(brain, nx, ny, initial_health=INIT_HEALTH, color=None)
            world.add_agent(agent)
            agents += 1

    return world


def create_smart_escape_world(**world_kwargs):
    """
    Injects "scent tracker" genes
    """
    world = GenerationalWorld(width=WIDTH, height=HEIGHT, cost_of_life=COST_OF_LIFE, **world_kwargs)

    agents = 0
    while agents < NUM_AGENTS:
        nx = random.randint(0, SPAWN_AREA_WIDTH - 1)
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

            agent = Agent(brain, nx, ny, initial_health=INIT_HEALTH, color=None)
            world.add_agent(agent)
            agents += 1

    return world

def create_two_island_world(**world_kwargs):
    world = TwoIslandWorld(width=WIDTH, height=HEIGHT, cost_of_life=COST_OF_LIFE, **world_kwargs)
    
    agents = 0
    while agents < NUM_AGENTS:
        nx = random.randint(0, SPAWN_AREA_WIDTH - 1)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            agent = Agent(brain, nx, ny, initial_health=INIT_HEALTH, color=None)
            world.add_agent(agent)
            agents += 1
            
    return world

def inject_migrator_genes(brain: Brain):
    """Incentivizes moving to scent. xAI will color them heavily GREEN"""
    brain.W1.fill(0.0)
    brain.b1.fill(0.0)
    brain.W2.fill(0.0)
    brain.b2.fill(0.0)
    for i in range(9):
        if i != 4:
            brain.W1[InputSchema.ID_SCENT + i, 0] = 1.0
    brain.W2[0, Action.MOVE_TO_SCENT] = 100.0
    brain.W2[1, Action.MOVE_RANDOM] = 5.0 # baseline

def inject_hunter_genes(brain: Brain):
    """Incentivizes taking from agents. xAI will color them heavily RED"""
    brain.W1.fill(0.0)
    brain.b1.fill(0.0)
    brain.W2.fill(0.0)
    brain.b2.fill(0.0)
    for i in range(9):
        if i != 4:
            brain.W1[InputSchema.ID_OTHER_HEALTH + i, 0] = 10.0
    brain.W2[0, Action.TAKE] = 100.0
    brain.W2[0, Action.MOVE_TOWARDS_AGENT] = 80.0
    brain.W2[1, Action.MOVE_RANDOM] = 10.0 # baseline

def create_competitive_world(**world_kwargs):
    # Higher cost of life forces Hunters to attack to survive
    world = CompetitiveWorld(width=WIDTH, height=HEIGHT, cost_of_life=COST_OF_LIFE_COMPETITIVE, **world_kwargs)
    
    # Spawn 40 Migrators
    migrators = 0
    while migrators < int(NUM_AGENTS * MIGRATORS_FRACTION):
        nx = random.randint(0, SPAWN_AREA_WIDTH - 1)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            inject_migrator_genes(brain)
            agent = Agent(brain, nx, ny, initial_health=INIT_HEALTH, color=None)
            # Attach faction tag for the evaluator
            agent.faction = "migrator" # type: ignore
            world.add_agent(agent)
            migrators += 1
            
    # Spawn 10 Hunters
    hunters = 0
    while hunters < int(NUM_AGENTS * (1 - MIGRATORS_FRACTION)):
        nx = random.randint(15, 20)
        ny = random.randint(0, HEIGHT - 1)
        if world.agent_grid[nx, ny] is None:
            brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
            inject_hunter_genes(brain)
            agent = Agent(brain, nx, ny, initial_health=INIT_HEALTH * HUNTERS_HEALTH_MULTIPLIER, color=None)
            # Attach faction tag for the evaluator
            agent.faction = "hunter" # type: ignore
            world.add_agent(agent)
            hunters += 1
            
    return world