import csv
import time
import os
import pickle
import numpy as np
from tqdm import tqdm

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.world_base import World

from ..scenarios.generational_scenarios import create_competitive_world, create_random_escape_world, create_smart_escape_world, create_two_island_world

WIDTH = 50
HEIGTH = 50
INIT_AGENTS = 50
INIT_HEALTH = 100.0
INIT_RES_DENSITY = 0.2
HIDDEN_DIM = 10

TOTAL_GENERATIONS = 2000
OUTPUT_FILE = "experiment_data.csv"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 100  # Save every 100 generations
MILESTONE_THRESHOLD = 0.95 # Save if an agent reaches 95% of the distance


SCENARIOS = {
    1: ("random", create_random_escape_world),
    2: ("smart", create_smart_escape_world),
    3: ("two-island", create_two_island_world),
    4: ("competitive", create_competitive_world),
}


def run_headless(world_creation_fun=create_random_escape_world, use_tqdm=True, output_file="experiment_data.csv"):
    print(f"Starting headless simulation. Target: {TOTAL_GENERATIONS} generations")

    # 1. Ensure our checkpoint folder exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    sim_start_time = time.time()
    world = world_creation_fun()
    data_buffer = []
    
    header = [
        "Generation", "Pop_Size", 
        "Max_Dist", "Avg_Dist", 
        "Max_Age", "Avg_Age", 
        "Avg_Health"
    ] + [f"Action_{a.name}" for a in Action]

    pbar = tqdm(total=TOTAL_GENERATIONS, desc="Simulating") if use_tqdm else None
    gen_action_counts = {action: 0 for action in Action}
    
    # Flag to tell tick 0 to save the children of a highly successful generation
    save_milestone_offspring = False 
    milestones_hit = 0 # To prevent saving a milestone every single generation after they master it

    while world.generation <= TOTAL_GENERATIONS:
        
        # --- NEW CHECKPOINTING LOGIC ---
        # We save at tick 0, so the loaded agents start fresh at the beginning of the epoch
        if world.tick == 0:
            is_interval = (world.generation % CHECKPOINT_INTERVAL == 0) or (world.generation == 1)
            
            if is_interval or save_milestone_offspring:
                label = "MILESTONE" if save_milestone_offspring and not is_interval else "INTERVAL"
                filepath = os.path.join(CHECKPOINT_DIR, f"gen_{world.generation:04d}_{label}.pkl")
                
                with open(filepath, "wb") as f:
                    # Pickle the agents list. (This saves their brains, weights, and initial state)
                    pickle.dump(world.agents, f)
                
                save_milestone_offspring = False
        # -------------------------------

        # --- CSV STATS LOGIC ---
        if world.tick == world.max_ticks - 1 or len(world.agents) == 0:
            pool = world.agents + getattr(world, 'graveyard', [])
            pop_size = len(pool)

            if pop_size > 0:
                max_dist = max([a.x for a in pool]) / world.width
                avg_dist = np.mean([a.x for a in pool]) / world.width
                max_age = max([a.age for a in pool])
                avg_age = np.mean([a.age for a in pool])
                avg_health = np.mean([max(0.0, a.health) for a in pool]) / 100.0 
            else:
                max_dist, avg_dist, max_age, avg_age, avg_health = 0.0, 0.0, 0, 0, 0.0

            # --- MILESTONE CHECK ---
            # If they broke the distance threshold, and we haven't already saved 5 milestones, flag it!
            if max_dist >= MILESTONE_THRESHOLD and milestones_hit < 5:
                save_milestone_offspring = True
                milestones_hit += 1

            action_list = [gen_action_counts[a] for a in Action]
            row = [
                world.generation, pop_size,
                f"{max_dist:.3f}", f"{avg_dist:.3f}",
                max_age, f"{avg_age:.1f}", f"{avg_health:.3f}"
            ] + action_list
            
            data_buffer.append(row)
            gen_action_counts = {action: 0 for action in Action}
            
            if pbar: pbar.update(1)

        # Advance the simulation
        world.update_world()
        
        for action in Action:
            gen_action_counts[action] += world.last_step_stats.get(action, 0)

    if pbar: pbar.close()

    sim_end_time = time.time()

    print(f"\nWriting {len(data_buffer)} observations to {output_file}")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_buffer)

    write_end_time = time.time()
    print(f"-- Simulation complete --")
    print(f"Simulation time: {sim_end_time - sim_start_time:.2f}s")
    print(f"Data write time: {write_end_time - sim_end_time:.2f}s")
    print(f"Data saved to {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run generational headless simulation")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=tuple(SCENARIOS.keys()),
        default=1,
        help="Scenario to use for the generational runner: 1=random, 2=smart, 3=two-island, 4=competitive",
    )
    args = parser.parse_args()

    _, world_creation_fun = SCENARIOS[args.scenario]
    run_headless(world_creation_fun=world_creation_fun)


if __name__ == "__main__":
    main()
