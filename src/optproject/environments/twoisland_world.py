import numpy as np
import random
from .generational_world import GenerationalWorld
from ..utils.nsga2 import get_twoisland_elites
from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.config import (
    MUTATION_PROB, MUTATION_AMP, ELITES_FRACTION, SCENT_SOURCE_STRENGTH,
    SCENT_INIT_DIFFUSION_STEPS, SCENT_DECAY, SCENT_DIFFUSION_RATE
)


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
        self.resource_grid[self.island_start_x :, 0 : self.heigth // 3] = self.max_resource # type: ignore
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = SCENT_SOURCE_STRENGTH
        
        # Island 2 (Bottom Right)
        self.resource_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.max_resource # type: ignore
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = SCENT_SOURCE_STRENGTH
        
        for _ in range(SCENT_INIT_DIFFUSION_STEPS):
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
        self.scent_grid = ((1.0 - SCENT_DIFFUSION_RATE) * s + SCENT_DIFFUSION_RATE * neigh_avg) * SCENT_DECAY

        # Re-apply strict island sources
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = SCENT_SOURCE_STRENGTH
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = SCENT_SOURCE_STRENGTH

    def resource_regrow(self, density: float = 0.05, amount: float = 5.0):
        """Override to regrow resources only on the two islands."""
        if amount <= 0:
            return
        mask = np.random.rand(self.width, self.heigth) < density
        island_mask = np.zeros_like(mask)
        island_mask[self.island_start_x :, 0 : self.heigth // 3] = True
        island_mask[self.island_start_x :, 2 * self.heigth // 3 :] = True
        mask = mask & island_mask
        self.resource_grid[mask] += amount
        self.resource_grid = np.minimum(self.resource_grid, self.max_resource)

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

            while len(self.agents) < self.initial_agent_count:
                nx = random.randint(0, 3)
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
                    agent = Agent(brain, nx, ny, initial_health=self.initial_health, color=None)
                    self.add_agent(agent)

        else:
            self._calculate_objectives(evaluation_pool)

            num_elites = int(ELITES_FRACTION * len(evaluation_pool))
            top_agents = get_twoisland_elites(evaluation_pool, num_elites)

            # REPRODUCTION
            self.agents = []
            self.agent_grid.fill(None)
            
            popsize = 0
            while popsize < self.initial_agent_count:
                parent = random.choice(top_agents)
                child = parent.reproduce(MUTATION_PROB, MUTATION_AMP, 0.0)

                nx = random.randint(0, 3)
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    child.x = nx
                    child.y = ny
                    child.health = self.initial_health
                    self.add_agent(child)
                    popsize += 1

        self.generation += 1
        self._init_epoch()