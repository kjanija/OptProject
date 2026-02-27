import random
import sys

import numpy as np

from agent import Action, Agent, InputSchema
from brain import Brain
from visualization import run_visualization
from world import World

WIDTH = 30
HEIGHT = 30
SEED = 999


class OasisWorld(World):
    """
    Custom World with regrow only in the center
    """

    def resource_regrow(self, density: float = 0.01, amount: float = 5.0):
        if amount <= 0:
            return
        mask = np.random.rand(self.width, self.heigth) < density

        cx, cy = self.width // 2, self.heigth // 2
        # zero mask everywhere except middle patch
        mask[: cx - 5, :] = False
        mask[cx + 5 :, :] = False
        mask[:, : cy - 5] = False
        mask[:, cy + 5 :] = False

        self.resource_grid[mask] += amount
        self.resource_grid = np.minimum(self.resource_grid, self.max_resource)


def inject_smart_genes(brain: Brain):
    """
    Injects the following strategy:
    1. If standing on food -> GATHER
    2. If food nearby -> MOVE_TO_RESOURCE
    3. Else -> MOVE_RANDOM
    """
    brain.W1.fill(0.0)
    brain.b1.fill(0.0)
    brain.W2.fill(0.0)
    brain.b2.fill(0.0)

    #############Hidden layer
    # Hidden neuron 0: standing on food
    brain.W1[InputSchema.ID_ENERGY + 4, 0] = 10.0

    # Hidden neuron 1: food nearby
    for i in range(9):
        if i != 4:
            brain.W1[InputSchema.ID_ENERGY + i, 1] = 1.0

    # Hidden neuron 2: always on bias
    brain.b1[2] = 1.0

    #############Output layer
    # strongly trigger GATHER if standing on food (h[0] high)
    brain.W2[0, Action.GATHER] = 100.0

    # strongly trigger MOVE_TO_RESOURCE if food nearby (h[1] high)
    brain.W2[1, Action.MOVE_TO_RESOURCE] = 50.0

    # baseline desire to MOVE_RANDOM (h[2] high)
    brain.W2[2, Action.MOVE_RANDOM] = 10.0


def create_deterministic_base(
    repro_threshold,
    repro_cost,
    maturity,
    cooldown,
    prob,
    world_class=World,
    regrow_density=0.01,
    regrow_amount=5.0,
):
    np.random.seed(SEED)
    random.seed(SEED)

    world = World(
        width=WIDTH,
        height=HEIGHT,
        cost_of_life=1.0,
        reproduction_threshold=repro_threshold,
        reproduction_cost=repro_cost,
        reproduction_maturity_age=maturity,
        reproduction_cooldown_time=cooldown,
        reproduction_prob=prob,
        regrow_density=regrow_density,
        regrow_amount=regrow_amount,
        init_res_density=0.0,  # start empty, will fill manually for various scenarios
    )

    # place agents in the center
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    positions = [
        (center_x, center_y),
        (center_x + 1, center_y),
        (center_x, center_y + 1),
        (center_x + 1, center_y + 1),
    ]
    for x, y in positions:
        brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
        inject_smart_genes(brain)  # smart genes = gatherer genes

        agent = Agent(
            brain, x, y, initial_health=70.0, color=None
        )  # init health set to trigger reproduction quickly
        world.add_agent(agent)

    return world


#################### Failure scenarios ###################################


def scenario_boomBust_petriDish():
    """
    Initially full of food, but zero regrow. Expect agents to rapidly
    consume the entire worlds resources and die afterwards
    """
    world = create_deterministic_base(41.0, 40.0, 0, 0, 1.0, regrow_amount=0.0)
    world.resource_grid.fill(world.max_resource)  # fill whole world once
    return world


def scenario_boomBust_oasis():
    """
    Food only present and "regrowable" in the middle. Expected that the
    overpopulation will quickly push (physically in search of an empty
    spot) the new agents out into the "desert"
    """
    world = create_deterministic_base(41.0, 40.0, 0, 0, 1.0, world_class=OasisWorld)
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource
    return world


def scenario_boomBust_apocalypse():
    """
    Food only in the middle and no regrow. An apocalyptic scenario in which
    the population collapses after an initial explosion in cardinality
    """
    world = create_deterministic_base(41.0, 40.0, 0, 0, 1.0, regrow_amount=0.0)
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource
    return world


##########################################################################


def scenario_maturity():
    """
    Expectation: agents must sutvive for a number of time steps before
    reproducing. So they explore first, population growth is delayed.
    """
    world = create_deterministic_base(
        repro_threshold=41.0,
        repro_cost=40.0,
        maturity=20,
        cooldown=0,
        prob=1.0,
        world_class=OasisWorld,
    )
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource
    return world


def scenario_cooldown():
    """
    Expectation: once an agent reproduces, he must wait a period before
    being able to reproduce again. They could leave a "trail" of children
    """
    world = create_deterministic_base(
        repro_threshold=41.0,
        repro_cost=40.0,
        maturity=0,
        cooldown=15,
        prob=1.0,
        world_class=OasisWorld,
    )
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource

    return world


def scenario_probabilistic():
    """
    Expectation: even if enough health, reproduction is probabilistic. Not
    everyone reproduces at once. Should be stable
    """
    world = create_deterministic_base(
        repro_threshold=41.0,
        repro_cost=40.0,
        maturity=0,
        cooldown=0,
        prob=0.1,
        world_class=OasisWorld,
    )
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource

    return world


def scenario_threshold():
    """
    Expectation: agents must have an enormous amount of health to reproduce
    should be a stable scenario
    """
    world = create_deterministic_base(
        repro_threshold=150.0,
        repro_cost=50.0,
        maturity=0,
        cooldown=0,
        prob=1.0,
        world_class=OasisWorld,
    )
    cx, cy = WIDTH // 2, HEIGHT // 2
    world.resource_grid[cx - 5 : cx + 5, cy - 5 : cy + 5] = world.max_resource

    for agent in world.agents:
        agent.health = 140.0

    return world


if __name__ == "__main__":
    print("=========================================")
    print("   REPRODUCTION MECHANICS SHOWCASE       ")
    print("=========================================")
    print("0A: The Petri Dish (Full grid, 0 Regrowth)")
    print("0B: The Oasis (Center patch, Normal Regrowth)")
    print("0C: The Apocalypse (Center patch, 0 Regrowth)")
    print(" 1: Maturity Age (Wait to grow up)")
    print(" 2: Cooldown Time (Wait between kids)")
    print(" 3: Probabilistic (Chance to reproduce)")
    print(" 4: High Threshold (Must be wealthy)")
    print("=========================================")

    choice = input("Enter selection (0A, 0B, 0C, 1, 2, 3, 4): ").strip().upper()

    scenarios = {
        "0A": scenario_boomBust_petriDish,
        "0B": scenario_boomBust_oasis,
        "0C": scenario_boomBust_apocalypse,
        "1": scenario_maturity,
        "2": scenario_cooldown,
        "3": scenario_probabilistic,
        "4": scenario_threshold,
    }

    if choice in scenarios:
        run_visualization(scenarios[choice])
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
