import random

import numpy as np

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.world_base import World
from ..utils.nsga2 import get_nsga2_elites
from ..core.config import (
    INIT_AGENTS, INIT_HEALTH, MAX_TICKS, ELITES_FRACTION,
    STORM_SPEED, ISLAND_WIDTH, MUTATION_PROB, MUTATION_AMP,
    SCENT_DECAY, SCENT_DIFFUSION_RATE, SCENT_SOURCE_STRENGTH,
    SCENT_DIFFUSION_STEPS, SCENT_INIT_DIFFUSION_STEPS
)


class GenerationalWorld(World):
    """
    Specialized world for generational evolution

    Includes a food island, "scent", moving storm and epoch-based reproduction
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.graveyard = []  # track dead agents for survival bias when ranking
        kwargs["regrow_amount"] = 0.0
        kwargs["reproduction_prob"] = 0.0
        kwargs["enable_scent_action"] = True
        
        self.initial_agent_count = INIT_AGENTS
        self.initial_health = INIT_HEALTH

        super().__init__(*args, **kwargs)

        self.generation = 1
        self.tick = 0
        self.max_ticks = MAX_TICKS

        self.island_start_x = self.width - ISLAND_WIDTH
        self.storm_speed = STORM_SPEED

        # init first epoch
        self._init_epoch()

    def _init_epoch(self):
        """Reset env for new generation"""
        self.tick = 0
        self.storm_x = -1
        self.scent_grid.fill(0.0)
        self.resource_grid.fill(0.0)
        self.graveyard.clear()

        self.resource_grid[self.island_start_x :, :] = self.max_resource
        self.scent_grid[self.island_start_x :, :] = SCENT_SOURCE_STRENGTH
        for _ in range(SCENT_INIT_DIFFUSION_STEPS):
            self.diffuse_scent()

    def remove_agent(self, agent):
        """
        Override to save dead agents for evaluation before removing them from 
        the grid
        """
        self.graveyard.append(agent)
        super().remove_agent(agent)

    def get_coords(self, x, y):
        """
        Override x-axis wrap around so agents don't cheat by walking left off
        screen to teleport to island on the right
        """
        nx = max(0, min(self.width - 1, int(x)))
        ny = int(y % self.heigth)  # Y wraps around
        return nx, ny

    def diffuse_scent(self):
        """
        Diffuse scent outwards from the island
        """

        s = self.scent_grid

        # y wraps
        s_up = np.roll(s, 1, axis=1)
        s_down = np.roll(s, -1, axis=1)

        # x is bounded
        s_left = np.zeros_like(s)
        s_left[:-1, :] = s[1:, :]
        s_right = np.zeros_like(s)
        s_right[1:, :] = s[:-1, :]

        # average neighbors and decay
        neigh_avg = (s_up + s_down + s_left + s_right) / 4.0
        self.scent_grid = ((1.0 - SCENT_DIFFUSION_RATE) * s + SCENT_DIFFUSION_RATE * neigh_avg) * SCENT_DECAY

        # scent at island is constant and strong
        self.scent_grid[self.island_start_x :, :] = SCENT_SOURCE_STRENGTH

    def __calculate_objectives(self, pool):
        """
        Calculates normalized distance and health as separate objectives
        """
        max_health = max([a.health for a in pool]) if pool else 1.0
        max_health = max(max_health, 1.0)
        max_x = float(self.width)
        max_age = float(self.max_ticks)

        for a in pool:
            a.norm_x =  a.x / max_x  # type: ignore
            a.norm_h = max(0, a.health) / max_health # type: ignore
            a.norm_age = a.age / max_age  # type: ignore

    def evaluate_n_evolve(self):

        evaluation_pool = self.agents + self.graveyard

        if len(evaluation_pool) < 2: # not enough agents for selection, etc..
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
            # FITNESS
            self.__calculate_objectives(evaluation_pool)

            # ELITISM (NSGA-II)
            top_agents = get_nsga2_elites(evaluation_pool, int(ELITES_FRACTION * len(evaluation_pool)))

            best_agent = top_agents[0]
            print(
                f"Gen {self.generation:3d} | Pool {len(evaluation_pool)} | Survivors {len(self.agents)} | "
                f"Best Front Rep -> Dist: {best_agent.norm_x:.2f} | " # type: ignore
                f"Health: {best_agent.norm_h:.2f} | " # type: ignore
                f"Age: {best_agent.norm_age:.2f}" # type: ignore
            )

            # reproduce
            self.agents = []
            self.agent_grid.fill(None)
            popsize = 0
            while popsize < self.initial_agent_count:
                parent = random.choice(top_agents)
                # cost is 0, manually reset health to max (reminder we are in a generational paradigm)
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

    def update_world(self):
        """Overrides standard step loop with generational logic"""
        self.tick += 1

        # move storm
        if self.tick % self.storm_speed == 0:
            if self.storm_x < self.island_start_x - 1:  # storm stops before island
                self.storm_x += 1

        # diffuse scent
        for _ in range(SCENT_DIFFUSION_STEPS):
            self.diffuse_scent()

        # standard agent steps (movement, action choice, dying to storm)
        super().update_world()

        # generational evaluation
        if self.tick >= self.max_ticks or not self.agents:
            self.evaluate_n_evolve()
