import random
import numpy as np
from agent import Agent, InputSchema, Action
from brain import Brain

class World:
    """
    Class to manage the whole environment (update loop, resources, etc...)
    """

    def __init__(
            self, 
            width: int, 
            height: int, 
            init_res_density: float = 0.1, 
            max_resource: float = 5.0,
            cost_of_life: float = 0.5,      # health to be removed at each round
            reproduction_threshold: float = 75.0,
            reproduction_cost = 50.0
        ):
        self.width = width
        self.heigth = height
        self.max_resource = max_resource
        self.cost_of_life = cost_of_life
        self.reproduction_threshold = reproduction_threshold
        self.reproduction_cost = reproduction_cost

        self.agents = []

        self.resource_grid = np.zeros((self.width, self.heigth))
        self.agent_grid = np.full((self.width, self.heigth), None, dtype=object)

        self.resource_regrow(init_res_density)

    def resource_regrow(self, density: float = 0.1, amount: float = 1.0):
        """Randomly add energy to the grid"""
        mask = np.random.rand(self.width, self.heigth) < density
        self.resource_grid[mask] += amount

        # Each cell has a "carying capacity" of resources, it cannot be infinite
        self.resource_grid = np.minimum(self.resource_grid, self.max_resource)

    def add_agent(self, agent: Agent = None):
        self.agents.append(agent)
        self.agent_grid[agent.x, agent.y] = agent

    def remove_agent(self, agent: Agent = None):
        if agent in self.agents:
            self.agents.remove(agent)
            self.agent_grid[agent.x, agent.y] = None

    def get_agent_inputs(self, agent):
        """Here we gather the 28-value input array used in the agent's brain"""
        inputs = np.zeros(InputSchema.TOTAL_INPUTS)
        c = 0 # 1d counter since inputs is a 1d vector

        # Here we scan the 3x3 neighbourhood
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # we do this since the grid is considered to be a torus
                nx, ny = (agent.x + dx) % self.width, (agent.y + dy) % self.heigth

                # energy
                inputs[InputSchema.ID_ENERGY + c] = self.resource_grid[nx, ny]

                # neighbouring agents
                neighbour = self.agent_grid[nx, ny]

                if neighbour is not None and neighbour is not agent:

                    # neighbouring health
                    inputs[InputSchema.ID_OTHER_HEALTH + c] = neighbour.health

                    # kinship/dna of neighbours
                    dist = np.linalg.norm(np.array(agent.color) - np.array(neighbour.color))
                    inputs[InputSchema.ID_OTHER_DNA + c] = max(0, 1.0 - dist)

                c += 1

        inputs[InputSchema.ID_SELF_HEALTH] = agent.health

        return inputs


    def execute_action(self, agent: Agent = None, action_idx: int = -1):
        """Execute action decided by agent"""

        action = Action(action_idx)

        def move_to(tx, ty):
            """move to target x and y"""
            if self.agent_grid[tx, ty] is None:
                self.agent_grid[agent.x, agent.y] = None
                self.agent_grid[tx, ty] = agent

                agent.x, agent.y = tx, ty
                return True # Successful movement
            return False
        
        # Let's manage the various actions now
        if action == Action.GATHER:
            energy = self.resource_grid[agent.x, agent.y]
            if energy > 0:
                agent.health += energy
                self.resource_grid[agent.x, agent.y] = 0

        # elif action == Action.WAIT:
            # pass # Right now does nothing, but in future I may change it

        elif action == Action.MOVE_RANDOM:
            dx, dy = random.choice([
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ])
            tx, ty = (agent.x + dx) & self.width, (agent.y + dy) & self.heigth
            move_to(tx, ty)

        elif action == Action.MOVE_TO_RESOURCE:
            best_res = -1
            best_pos = None
        
            # scan the neighbourhood for the cell with max energy
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0: continue # since we MOVE to resource
                    tx, ty = (agent.x + dx) & self.width, (agent.y + dy) & self.heigth

                    if self.resource_grid[tx, ty] > best_res and self.agent_grid[tx, ty] is None:
                        best_res = self.resource_grid[tx, ty]
                        best_pos = (tx, ty)

            # move to the cell with max energy, if it exists
            if best_pos:
                move_to(*best_pos)

            elif action == Action.MOVE_AWAY_FROM_AGENT:
                pass
                # TODO: un primo modo sarebbe muovere in una cella senza vicini

            elif action == Action.MOVE_TOWARDS_AGENT:
                pass
                # TODO

            elif action == Action.GIVE:
                pass
                # TODO

            elif action == Action.TAKE:
                pass 
                # TODO

    def step_agents(self):
        """Each agent takes a step"""

        # For now I avoid complex "collision" resolution and just shuffle at
        # every round the order of the agents as to reduce the order bias
        random.shuffle(self.agents)

        for agent in self.agents[:]:
            # we copy the list because we are making changes to it inside the loop

            # get info from env
            inputs = self.get_agent_inputs(agent)
            
            # think
            action_id = agent.brain.predict(inputs)

            # act
            self.execute_action(agent, action_id)

            # cost of living
            agent.health -= self.cost_of_life

            if agent.health <= 0:
                self.remove_agent(agent)
                continue # if dead, can't reproduce

            # reproduce if possible
            if agent.health >= self.reproduction_cost:
                # search for a spot
                found = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0: continue
                        tx, ty = (agent.x + dx) & self.width, (agent.y + dy) & self.heigth
                        if self.agent_grid[tx, ty] is None:
                            # spot found
                            child = agent.reproduce(0.05, 0.1, self.reproduction_cost)
                            child.x, child.y = tx, ty
                            self.add_agent(child)
                            found = True
                            break
                    if found:
                        break


    def update_world(self):
        """
        The main simulation step
        """
        self.resource_regrow()
        self.step_agents()

if __name__ == "__main__":

    print("#"*50)
    print("\n--- Testing World Class ---")
    WORLD_WIDTH = 20
    WORLD_HEIGHT = 20
    
    world = World(WORLD_WIDTH, WORLD_HEIGHT)
    print("World initialized.")
    
    # Create an agent and add it
    brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
    agent = Agent(brain, 5, 5, 50.0, (1,0,0))
    world.add_agent(agent)
    
    print(f"Agent added at {agent.x}, {agent.y}")
    assert world.agent_grid[5, 5] == agent
    
    # Run some steps
    for i in range(10):
        print(f"Step: {i+1}--------------------------")
        world.update_world()
        print("World stepped successfully.")
        print(f"Agent new health (approx 49.5): {agent.health}")
        
        # Check if agent moved or stayed (depends on random brain init)
        print(f"Agent post-step position: {agent.x}, {agent.y}")
    
    print("\nAll systems operational.")