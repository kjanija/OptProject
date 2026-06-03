import random

import numpy as np

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.world_base import World


class GenerationalWorld(World):
    """
    Specialized world for generational evolution

    Includes a food island, "scent", moving storm and epoch-based reproduction
    """

    def __init__(
        self,
        scent_decay=0.98,
        scent_diffusion_rate=0.35,
        scent_source_strength=100.0,
        scent_diffusion_steps=3,
        scent_init_diffusion_steps=150,
        *args,
        **kwargs,
    ):
        # disable automatic regrowth and continuous reproduction
        self.scent_decay = scent_decay  # persistence (->1 => longer persistence)
        self.scent_diffusion_rate = scent_diffusion_rate  # spread speed (higher => faster)
        self.scent_source_strength = scent_source_strength  # source intensity
        self.scent_diffusion_steps = scent_diffusion_steps  # higher => more spread per tick
        self.scent_init_diffusion_steps = scent_init_diffusion_steps
        kwargs["regrow_amount"] = 0.0
        kwargs["reproduction_prob"] = 0.0
        kwargs["enable_scent_action"] = True

        super().__init__(*args, **kwargs)

        self.generation = 1
        self.tick = 0
        self.max_ticks = 200

        self.island_start_x = self.width - 5
        self.storm_speed = 10  # storm moves 1 unit every 10 ticks

        # init first epoch
        self._init_epoch()

    def _init_epoch(self):
        """Reset env for new generation"""
        self.tick = 0
        self.storm_x = -1
        self.scent_grid.fill(0.0)
        self.resource_grid.fill(0.0)

        self.resource_grid[self.island_start_x :, :] = self.max_resource
        self.scent_grid[self.island_start_x :, :] = self.scent_source_strength
        for _ in range(self.scent_init_diffusion_steps):
            self.diffuse_scent()

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
        self.scent_grid = (
            (1.0 - self.scent_diffusion_rate) * s
            + self.scent_diffusion_rate * neigh_avg
        ) * self.scent_decay

        # scent at island is constant and strong
        self.scent_grid[self.island_start_x :, :] = self.scent_source_strength

    def evaluate_n_evolve(self):

        if not self.agents:
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
            # FITNESS
            max_health = max([a.health for a in self.agents]) if self.agents else 1.0
            max_health = max(max_health, 1.0)
            max_x = float(self.width)

            for a in self.agents:
                norm_x = a.x / max_x
                norm_h = a.health / max_health
                a.fitness = (norm_x + norm_h) / 2.0

            self.agents.sort(key=lambda a: a.fitness, reverse=True)

            best_agent = self.agents[0]
            print(
                f"Gen {self.generation} | Survivors {len(self.agents)} | fittest Distance {max_x} | fittest health {best_agent.health:.1f} | Fitest: {best_agent.fitness:.3f}"
            )

            # ELITISM
            top_agents = self.agents[: min(10, len(self.agents))]

            # reproduce
            self.agents = []
            self.agent_grid.fill(None)
            popsize = 0
            while popsize < 50:
                parent = random.choice(top_agents)
                # cost is 0, manually reset health to max (reminder we are in a generational paradigm)
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

    def update_world(self):
        """Overrides standard step loop with generational logic"""
        self.tick += 1

        # move storm
        if self.tick % self.storm_speed == 0:
            if self.storm_x < self.island_start_x - 1:  # storm stops before island
                self.storm_x += 1

        # diffuse scent
        for _ in range(self.scent_diffusion_steps):
            self.diffuse_scent()

        # standard agent steps (movement, action choice, dying to storm)
        super().update_world()

        # generational evaluation
        if self.tick >= self.max_ticks or not self.agents:
            self.evaluate_n_evolve()
