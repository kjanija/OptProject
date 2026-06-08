import numpy as np
import random
from .generational_world import GenerationalWorld
from ..utils.nsga2 import get_twoisland_elites
from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema


class TwoIslandWorld(GenerationalWorld):
    """
    Splits the survival objective into two physically distant locations 
    to test Diversity Maintenance (Speciation).
    """
    def _init_epoch(self):
        """Override to spawn TWO islands instead of one big one"""
        self.tick = 0
        self.storm_x = -1
        self.scent_grid.fill(0.0)
        self.resource_grid.fill(0.0)
        if hasattr(self, 'graveyard'):
            self.graveyard.clear()

        # island centers for objective mapping
        self.island_1_y = self.heigth // 6
        self.island_2_y = 5 * self.heigth // 6

        # Island 1 (Top Right)
        self.resource_grid[self.island_start_x :, 0 : self.heigth // 3] = self.max_resource
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = self.scent_source_strength
        
        # Island 2 (Bottom Right)
        self.resource_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.max_resource
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.scent_source_strength
        
        for _ in range(self.scent_init_diffusion_steps):
            self.diffuse_scent()

    def get_coords(self, x, y):
        """
        Flat grid, no wrapping, so that we actually have two islands
        """
        nx = max(0, min(self.width - 1, int(x)))
        ny = max(0, min(self.heigth - 1, int(y)))
        return nx, ny

    def diffuse_scent(self):
        """Override to maintain the two distinct scent sources that do not
        wrap around the world and blend together."""
        s = self.scent_grid

        # Shift arrays with zero-padding instead of np.roll (which wraps)
        s_up = np.zeros_like(s)
        s_up[:, :-1] = s[:, 1:]
        
        s_down = np.zeros_like(s)
        s_down[:, 1:] = s[:, :-1]

        s_left = np.zeros_like(s)
        s_left[:-1, :] = s[1:, :]
        
        s_right = np.zeros_like(s)
        s_right[1:, :] = s[:-1, :]

        neigh_avg = (s_up + s_down + s_left + s_right) / 4.0
        self.scent_grid = (
            (1.0 - self.scent_diffusion_rate) * s
            + self.scent_diffusion_rate * neigh_avg
        ) * self.scent_decay

        # Re-apply strict island sources
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = self.scent_source_strength
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.scent_source_strength

    def _calculate_objectives(self, pool):
        """
        Helper method to calculate and assign normalized spatial objectives
        """
        max_health = max([a.health for a in pool]) if pool else 1.0
        max_health = max(max_health, 1.0)
        
        # Max possible distance on the grid (diagonal)
        max_dist = np.sqrt(self.width**2 + self.heigth**2) 

        for a in pool:
            # Calculate physical distance to the center of both islands
            dist_1 = np.sqrt((a.x - self.width)**2 + (a.y - self.island_1_y)**2)
            dist_2 = np.sqrt((a.x - self.width)**2 + (a.y - self.island_2_y)**2)

            # Convert distance to a proximity score (higher is better)
            a.norm_prox_1 = 1.0 - (dist_1 / max_dist)
            a.norm_prox_2 = 1.0 - (dist_2 / max_dist)
            a.norm_h = max(0.0, a.health) / max_health

    def evaluate_n_evolve(self):
        """
        We change the spatial objective to be two competing objectives of 
        distance to islands
        """
        evaluation_pool = self.agents + getattr(self, 'graveyard', [])

        if len(evaluation_pool) < 2:
            print(f"Gen {self.generation}: Extinction. Respawning random mutants")
            self.agents = []
            self.agent_grid.fill(None)

            while len(self.agents) < 50:
                nx = random.randint(0, 3)
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
                    agent = Agent(brain, nx, ny, initial_health=100.0, color=None)
                    self.add_agent(agent)

        else:
            self._calculate_objectives(evaluation_pool)

            top_agents = get_twoisland_elites(evaluation_pool, 10)

            # Log the stats of the absolute best representative from Rank 1
            best_agent = top_agents[0]
            print(
                f"Gen {self.generation:3d} | Pool {len(evaluation_pool)} | Survivors {len(self.agents)} | "
                f"Best Front Rep -> Prox 1: {best_agent.norm_prox_1:.2f} | " # type: ignore
                f"Prox 2: {best_agent.norm_prox_2:.2f} | " # type: ignore
                f"Health: {best_agent.norm_h:.2f}" # type: ignore
            )

            # 3. REPRODUCTION
            self.agents = []
            self.agent_grid.fill(None)
            
            popsize = 0
            while popsize < 50:
                parent = random.choice(top_agents)
                child = parent.reproduce(
                    mutation_prob=0.1, mutation_amp=0.2, reproduction_cost=0.0
                )

                nx = random.randint(0, 3)
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    child.x = nx
                    child.y = ny
                    child.health = 100.0
                    self.add_agent(child)
                    popsize += 1

        self.generation += 1
        self._init_epoch()