import csv
import time

import numpy as np
from tqdm import tqdm

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.world_base import World

WIDTH = 50
HEIGTH = 50
INIT_AGENTS = 50
INIT_HEALTH = 100.0
INIT_RES_DENSITY = 0.2
HIDDEN_DIM = 10

TOTAL_GENERATIONS = 2000
OUTPUT_FILE = "experiment_data.csv"


def create_world():
    world = World(WIDTH, HEIGTH, init_res_density=INIT_RES_DENSITY)

    input_dim = InputSchema.TOTAL_INPUTS
    hidden_dim = HIDDEN_DIM
    output_dim = len(Action)

    popsize = 0
    while popsize < INIT_AGENTS:
        brain = Brain(
            input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim
        )
        x = np.random.randint(0, WIDTH)
        y = np.random.randint(0, HEIGTH)
        color = None

        agent = Agent(brain, x, y, initial_health=INIT_HEALTH, color=color)
        if world.add_agent(agent=agent):
            popsize += 1

    return world


def run_headless(world_creation_fun=create_world, use_tqdm=True, output_file=OUTPUT_FILE):
    """
    Runs the simulation in headless mode (faster) and saves the results in a csv file

    Args:
        world_creation_fun: Function that returns a configured World object
        output_file: Path where the CSV will be saved
    """
    print(f"Starting headless simulation. Total generations: {TOTAL_GENERATIONS}")

    sim_start_time = time.time()
    world = world_creation_fun()

    data_buffer = []
    header = ["Generation", "Population", "Avg_health"] + [a.name for a in Action]

    if use_tqdm:
        try:
            iterator = tqdm(range(TOTAL_GENERATIONS), desc="Simulating")
        except ImportError:
            print("tqdm error, maybe it's missing")
            iterator = range(TOTAL_GENERATIONS)
    else:
        iterator = range(TOTAL_GENERATIONS)

    for gen in iterator:
        world.update_world()

        pop_size = len(world.agents)

        if pop_size > 0:
            avg_health = np.mean([a.health for a in world.agents])
        else:
            avg_health = 0.0
            if gen > 0 and pop_size == 0:
                print(f"Extinction at generation {gen}")
                action_counts = [0] * len(Action)
                row = [gen, pop_size, avg_health] + action_counts
                data_buffer.append(row)
                break

        action_counts = [world.last_step_stats.get(action, 0) for action in Action]
        row = [gen, pop_size, avg_health] + action_counts
        data_buffer.append(row)

    sim_end_time = time.time()

    print(f"Writing {len(data_buffer)} observations to {output_file}")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_buffer)

    write_end_time = time.time()
    print(f"-- Simulation complete --")
    print(f"Simulation time: {sim_end_time - sim_start_time}")
    print(f"Data write time: {write_end_time - sim_end_time}")
    print(f"Data saved to {output_file}")


def main():
    run_headless()


if __name__ == "__main__":
    main()
