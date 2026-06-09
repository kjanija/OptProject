import random

import numpy as np

from .actions import Action
from .agent import Agent
from .brain import Brain
from .schema import InputSchema
from .config import (
    MAX_RESOURCE, COST_OF_LIFE, REPRODUCTION_THRESHOLD, REPRODUCTION_COST,
    REPRODUCTION_MATURITY_AGE, REPRODUCTION_COOLDOWN, REPRODUCTION_PROB,
    REGROW_DENSITY, REGROW_AMOUNT, GIVE_AMOUNT, TAKE_AMOUNT, INIT_RES_DENSITY,
    TAKE_COST_FRAC
)


class World:
    """
    Class to manage the whole environment (update loop, resources, etc...)
    """

    def __init__(
        self,
        width: int,
        height: int,
        init_res_density: float = INIT_RES_DENSITY,
        max_resource: float = MAX_RESOURCE,
        cost_of_life: float = COST_OF_LIFE,
        reproduction_threshold: float = REPRODUCTION_THRESHOLD,
        reproduction_cost=REPRODUCTION_COST,
        reproduction_maturity_age: int = REPRODUCTION_MATURITY_AGE,
        reproduction_cooldown_time: int = REPRODUCTION_COOLDOWN,
        reproduction_prob: float = REPRODUCTION_PROB,
        regrow_density: float = REGROW_DENSITY,
        regrow_amount: float = REGROW_AMOUNT,
        enable_scent_action: bool = False,
    ):
        self.width = width
        self.heigth = height
        self.max_resource = max_resource
        self.cost_of_life = cost_of_life

        self.reproduction_threshold = reproduction_threshold
        self.reproduction_cost = reproduction_cost
        self.reproduction_maturity_age = reproduction_maturity_age
        self.reproduction_cooldown_time = reproduction_cooldown_time
        self.reproduction_prob = reproduction_prob

        self.regrow_density = regrow_density
        self.regrow_amount = regrow_amount

        self.enable_scent_action = enable_scent_action

        self.agents: list[Agent] = []
        self.resource_grid = np.zeros((self.width, self.heigth))
        self.agent_grid = np.full((self.width, self.heigth), None, dtype=object)
        self.last_step_stats = {action: 0 for action in Action}

        self.scent_grid = np.zeros((self.width, self.heigth))
        self.storm_x = -1  # initial position of deadly storm

        self.resource_regrow(density=init_res_density, amount=10.0)

    def resource_regrow(self, density: float = 0.05, amount: float = 5.0):
        """Randomly add energy to the grid"""
        if amount <= 0:
            return
        mask = np.random.rand(self.width, self.heigth) < density
        self.resource_grid[mask] += amount

        # Each cell has a "carying capacity" of resources, it cannot be infinite
        self.resource_grid = np.minimum(self.resource_grid, self.max_resource)

    def add_agent(self, agent: Agent):
        if self.agent_grid[agent.x, agent.y] is not None:
            return False  # occupied cell
        self.agents.append(agent)
        self.agent_grid[agent.x, agent.y] = agent
        return True

    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)
            self.agent_grid[agent.x, agent.y] = None

    def get_coords(self, x, y):
        """Returns the eventually wrapped coods"""
        # we do this since the grid is considered to be a torus
        return int(x % self.width), int(y % self.heigth)

    def get_agent_inputs(self, agent):
        """Here we gather the InputSchema.TOTAL_INPUTS-value input array used in the agent's brain"""
        inputs = np.zeros(InputSchema.TOTAL_INPUTS)
        c = 0  # 1d counter since inputs is a 1d vector

        # Here we scan the 3x3 neighbourhood
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # we do this since the grid is considered to be a torus
                nx, ny = self.get_coords(agent.x + dx, agent.y + dy)

                # energy
                inputs[InputSchema.ID_ENERGY + c] = self.resource_grid[nx, ny]
                inputs[InputSchema.ID_SCENT + c] = self.scent_grid[nx, ny]

                # neighbouring agents
                neighbour = self.agent_grid[nx, ny]

                if neighbour is not None and neighbour is not agent:

                    # neighbouring health
                    inputs[InputSchema.ID_OTHER_HEALTH + c] = neighbour.health

                    # kinship/dna of neighbours
                    dist = np.linalg.norm(
                        np.array(agent.color) - np.array(neighbour.color)
                    )
                    inputs[InputSchema.ID_OTHER_DNA + c] = max(0, 1.0 - float(dist))

                c += 1

        inputs[InputSchema.ID_SELF_HEALTH] = agent.health

        return inputs

    def execute_action(self, agent: Agent, inputs: np.ndarray, action_idx: int = -1):
        """Execute action decided by agent"""

        action = Action(action_idx)

        if action == Action.MOVE_TO_SCENT and not self.enable_scent_action:
            action = Action.WAIT

        # log action
        self.last_step_stats[action] += 1

        def move_to(tx, ty):
            """move to target x and y"""
            # For debug purposes : print(f"Target (x,y)={tx, ty}")
            if self.agent_grid[tx, ty] is None:
                self.agent_grid[agent.x, agent.y] = None
                self.agent_grid[tx, ty] = agent

                agent.x, agent.y = tx, ty
                return True  # Successful movement
            return False

        def get_neighbor_coords(dx, dy):
            return self.get_coords(agent.x + dx, agent.y + dy)

        # Let's manage the various actions now
        if action == Action.GATHER:
            energy = self.resource_grid[agent.x, agent.y]
            if energy > 0:
                agent.health += energy
                self.resource_grid[agent.x, agent.y] = 0

        elif action == Action.WAIT:
            pass  # Right now does nothing, but in future I may change it

        elif action == Action.MOVE_RANDOM:
            dx, dy = random.choice(
                [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            )
            tx, ty = get_neighbor_coords(dx, dy)
            move_to(tx, ty)

        elif action == Action.MOVE_TO_RESOURCE:

            neighbourhood_energy = inputs[
                InputSchema.ID_ENERGY : InputSchema.ID_ENERGY + 9
            ]
            max_energy = np.max(neighbourhood_energy)

            if max_energy > 0:
                best_indices = np.where(neighbourhood_energy == max_energy)[0]
                best_indices = best_indices[best_indices != 4] # because idx 4 is self
                if len(best_indices) > 0:
                    best_idx = np.random.choice(best_indices)
                    # go back to a 2d type indicization
                    dy = int(best_idx // 3 - 1)
                    dx = int(best_idx % 3 - 1)

                    tx, ty = get_neighbor_coords(dx, dy)
                    move_to(tx, ty)

        elif action == Action.MOVE_TO_SCENT:
            neighborhood_scent = inputs[InputSchema.ID_SCENT : InputSchema.ID_SCENT + 9]
            max_scent = np.max(neighborhood_scent)
            
            if max_scent > 0:
                best_indices = np.where(neighborhood_scent == max_scent)[0]
                best_indices = best_indices[best_indices != 4]
                if len(best_indices) > 0:
                    best_idx = np.random.choice(best_indices)
                    dy = int(best_idx // 3 - 1)
                    dx = int(best_idx % 3 - 1)

                    tx, ty = get_neighbor_coords(dx, dy)
                    move_to(tx, ty)

        elif action == Action.MOVE_AWAY_FROM_AGENT:
            # will use agents health as an indicator of presence
            neighbourhood_health = inputs[
                InputSchema.ID_OTHER_HEALTH : InputSchema.ID_OTHER_HEALTH + 9
            ]
            empty_spots = []
            for i in range(9):
                if i == 4:
                    continue
                if neighbourhood_health[i] == 0:
                    empty_spots.append(i)

            if empty_spots:
                chosen = random.choice(empty_spots)
                dy = int(chosen // 3 - 1)
                dx = int(chosen % 3 - 1)
                tx, ty = get_neighbor_coords(dx, dy)
                move_to(tx, ty)
            else:
                # random move if surrounded
                dx, dy = random.choice(
                    [
                        (0, 1),
                        (0, -1),
                        (1, 0),
                        (-1, 0),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ]
                )
                tx, ty = get_neighbor_coords(dx, dy)
                move_to(tx, ty)

        elif action == Action.MOVE_TOWARDS_AGENT:
            # will use still health as indicator of presence
            neighbourhood_health = inputs[
                InputSchema.ID_OTHER_HEALTH : InputSchema.ID_OTHER_HEALTH + 9
            ]
            max_health = np.max(neighbourhood_health)

            if max_health > 0:
                best_indices = np.where(neighbourhood_health == max_health)[0]
                best_indices = best_indices[best_indices != 4]
                if len(best_indices) > 0:
                    best_idx = np.random.choice(best_indices)
                    dy = int(best_idx // 3 - 1)
                    dx = int(best_idx % 3 - 1)
                    tx, ty = get_neighbor_coords(dx, dy)
                    move_to(tx, ty)

        elif action == Action.GIVE:
            # give 10 health to a random neighbor
            neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    tx, ty = get_neighbor_coords(dx, dy)
                    n = self.agent_grid[tx, ty]
                    if n:
                        neighbors.append(n)

            if neighbors and agent.health > 10:
                receiver = random.choice(neighbors)
                agent.health -= GIVE_AMOUNT
                receiver.health += GIVE_AMOUNT

        elif action == Action.TAKE:
            # steal 10 health from a neighbor, but this has a cost to the attacker as well
            neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    tx, ty = get_neighbor_coords(dx, dy)
                    n = self.agent_grid[tx, ty]
                    if n:
                        neighbors.append(n)

            if neighbors:
                victim = random.choice(neighbors)
                amount = min(TAKE_AMOUNT, victim.health)
                victim.health -= amount
                agent.health += amount*(1 - TAKE_COST_FRAC)  # attacker gets less due to cost

    def can_reproduce(self, agent: Agent) -> bool:
        if agent.health <= self.reproduction_threshold:
            return False
        if agent.age < self.reproduction_maturity_age:
            return False
        if agent.reproduction_cooldown > 0:
            return False
        if random.random() > self.reproduction_prob:
            return False

        return True

    def step_agents(self):
        """Each agent takes a step"""

        # Reset the stats
        self.last_step_stats = {action: 0 for action in Action}

        # For now I avoid complex "collision" resolution and just shuffle at
        # every round the order of the agents as to reduce the order bias
        random.shuffle(self.agents)

        for agent in self.agents[:]:
            # we copy the list because we are making changes to it inside the loop

            agent.age += 1
            if agent.reproduction_cooldown > 0:
                agent.reproduction_cooldown -= 1

            # act
            inputs = self.get_agent_inputs(agent)  # get info from env
            action_id = agent.brain.predict(inputs)  # think
            self.execute_action(agent, inputs, int(action_id))  # act

            # cost of living
            agent.health -= self.cost_of_life

            if agent.x <= self.storm_x:
                agent.health = 0.0

            if agent.health <= 0:
                self.remove_agent(agent)
                continue  # if dead, can't reproduce

            # reproduce if possible
            if self.can_reproduce(agent):
                # search for a spot
                found = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        tx, ty = self.get_coords(agent.x + dx, agent.y + dy)
                        if self.agent_grid[tx, ty] is None:
                            # spot found
                            child = agent.reproduce(0.05, 0.1, self.reproduction_cost)
                            child.x, child.y = tx, ty
                            self.add_agent(child)
                            agent.reproduction_cooldown = (
                                self.reproduction_cooldown_time
                            )
                            found = True
                            break
                    if found:
                        break

    def update_world(self):
        """
        The main simulation step
        """
        self.resource_regrow(density=self.regrow_density, amount=self.regrow_amount)
        self.step_agents()


def run_world_self_tests():
    print("\n=== STARTING COMPREHENSIVE UNIT TESTS ===\n")

    # Helper function to reset world
    def create_test_world():
        w = World(10, 10, init_res_density=0.0)  # Empty world
        # Clear any random resources added by init
        w.resource_grid.fill(0)
        return w

    # Helper to create a dummy agent
    def create_dummy_agent(x, y, health=100.0):
        brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
        return Agent(brain, x, y, health, (1, 0, 0))

    # --- TEST 1: GATHER ---
    print("Test 1: GATHER Action...", end=" ")
    world = create_test_world()
    agent = create_dummy_agent(5, 5, health=50.0)
    world.add_agent(agent)
    world.resource_grid[5, 5] = 10.0
    inputs = world.get_agent_inputs(agent)
    world.execute_action(agent, inputs, Action.GATHER)
    assert agent.health == 60.0, f"Health should be 60, got {agent.health}"
    assert world.resource_grid[5, 5] == 0.0, "Resource should be consumed"
    print("PASSED")

    # --- TEST 2: WAIT ---
    print("Test 2: WAIT Action...", end=" ")
    world = create_test_world()
    agent = create_dummy_agent(5, 5, health=50.0)
    world.add_agent(agent)
    inputs = world.get_agent_inputs(agent)
    world.execute_action(agent, inputs, Action.WAIT)
    assert agent.health == 50.0, "Health should not change during WAIT"
    assert agent.x == 5 and agent.y == 5, "Agent should not move"
    print("PASSED")

    # --- TEST 3: MOVE_TO_RESOURCE ---
    print("Test 3: MOVE_TO_RESOURCE...", end=" ")
    world = create_test_world()
    agent = create_dummy_agent(5, 5)
    world.add_agent(agent)
    world.resource_grid[6, 5] = 10.0  # Right
    world.resource_grid[4, 5] = 0.0  # Left
    inputs = world.get_agent_inputs(agent)
    world.execute_action(agent, inputs, Action.MOVE_TO_RESOURCE)
    assert (
        agent.x == 6 and agent.y == 5
    ), f"Agent should move to (6, 5), got ({agent.x}, {agent.y})"
    assert world.agent_grid[5, 5] is None, "Old spot should be empty"
    assert world.agent_grid[6, 5] == agent, "New spot should be occupied"
    print("PASSED")

    # --- TEST 4: GIVE (Altruism) ---
    print("Test 4: GIVE Action...", end=" ")
    world = create_test_world()
    giver = create_dummy_agent(5, 5, health=50.0)
    receiver = create_dummy_agent(5, 6, health=20.0)
    world.add_agent(giver)
    world.add_agent(receiver)
    inputs = world.get_agent_inputs(giver)
    world.execute_action(giver, inputs, Action.GIVE)
    assert giver.health == 40.0, f"Giver should lose 10, got {giver.health}"
    assert receiver.health == 30.0, f"Receiver should gain 10, got {receiver.health}"
    print("PASSED")

    # --- TEST 5: TAKE (Aggression) ---
    print("Test 5: TAKE Action...", end=" ")
    world = create_test_world()
    attacker = create_dummy_agent(5, 5, health=50.0)
    victim = create_dummy_agent(5, 6, health=20.0)
    world.add_agent(attacker)
    world.add_agent(victim)
    inputs = world.get_agent_inputs(attacker)
    world.execute_action(attacker, inputs, Action.TAKE)
    assert victim.health == 10.0, f"Victim should lose 10, got {victim.health}"
    assert attacker.health == 60.0, f"Attacker should gain 10, got {attacker.health}"
    print("PASSED")

    # --- TEST 6: MOVE_AWAY_FROM_AGENT ---
    print("Test 6: MOVE_AWAY_FROM_AGENT...", end=" ")
    world = create_test_world()
    fleeing_agent = create_dummy_agent(5, 5)
    scary_neighbor = create_dummy_agent(5, 6)  # To the right
    world.add_agent(fleeing_agent)
    world.add_agent(scary_neighbor)
    inputs = world.get_agent_inputs(fleeing_agent)
    world.execute_action(fleeing_agent, inputs, Action.MOVE_AWAY_FROM_AGENT)
    assert (fleeing_agent.x, fleeing_agent.y) != (5, 6), "Should not move into neighbor"
    assert (fleeing_agent.x, fleeing_agent.y) != (5, 5), "Should have moved"
    print("PASSED")

    # --- TEST 7: MOVE_TOWARDS_AGENT (Bump test) ---
    print("Test 7: MOVE_TOWARDS_AGENT...", end=" ")
    world = create_test_world()
    stalker = create_dummy_agent(5, 5)
    target = create_dummy_agent(6, 5)
    world.add_agent(stalker)
    world.add_agent(target)
    inputs = world.get_agent_inputs(stalker)
    world.execute_action(stalker, inputs, Action.MOVE_TOWARDS_AGENT)
    # Logic implies checking neighbor, finding max health, trying to move there, failing.
    assert stalker.x == 5 and stalker.y == 5, "Should stay put (collision)"
    print("PASSED")

    # --- TEST 8: MOVE_RANDOM ---
    print("Test 8: MOVE_RANDOM...", end=" ")
    world = create_test_world()
    agent = create_dummy_agent(5, 5)
    world.add_agent(agent)
    initial_pos = (agent.x, agent.y)
    moved = False
    for _ in range(10):
        inputs = world.get_agent_inputs(agent)
        world.execute_action(agent, inputs, Action.MOVE_RANDOM)
        if (agent.x, agent.y) != initial_pos:
            moved = True
            break
    assert moved, "Agent should eventually move randomly"
    print("PASSED")

    # --- TEST 9: REPRODUCTION ---
    print("Test 9: Reproduction...", end=" ")
    world = create_test_world()
    parent = create_dummy_agent(5, 5, health=90.0)
    world.add_agent(parent)
    world.step_agents()
    assert (
        parent.health < 60.0
    ), f"Parent should have paid cost. Health: {parent.health}"
    assert len(world.agents) == 2, "Population should be 2"
    print("PASSED")

    print("\n=== ALL UNIT TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_world_self_tests()
