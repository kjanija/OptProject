import os
import glob
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.environments.generational_world import GenerationalWorld
from optproject.runners.visualization_runner import run_visualization
from optproject.scenarios.generational_scenarios import (
    create_random_escape_world,
    create_smart_escape_world,
    create_two_island_world,
    create_competitive_world,
)
from optproject.core.config import WIDTH, HEIGTH, INIT_RES_DENSITY, CHECKPOINT_DIR

# Ensure these match the constants used during training

SCENARIOS = {
    "1": ("random", create_random_escape_world),
    "2": ("smart", create_smart_escape_world),
    "3": ("two-island", create_two_island_world),
    "4": ("competitive", create_competitive_world),
}

def get_latest_checkpoint(scenario_name=None):
    """Finds the most recently modified pickle file, optionally within a scenario."""
    search_dir = CHECKPOINT_DIR
    if scenario_name:
        search_dir = os.path.join(CHECKPOINT_DIR, scenario_name)

    if not os.path.exists(search_dir):
        return None
    
    search_path = os.path.join(search_dir, "**", "*.pkl")
    files = glob.glob(search_path, recursive=True)
    if not files:
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_checkpoint_world(checkpoint_path, world_creation_fn=None):
    """
    This acts as our custom 'world_creation_fun'. 
    It loads the agents, builds the world, and returns it.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        saved_agents = pickle.load(f)

    print(f"Loaded {len(saved_agents)} trained agents. Initializing world...")

    # 1. Create a fresh Generational World
    if world_creation_fn:
        world = world_creation_fn()
    else:
        world = GenerationalWorld(width=WIDTH, height=HEIGTH, init_res_density=INIT_RES_DENSITY)
    
    # 2. Clear out whatever random agents it spawned by default
    world.agents = []
    world.agent_grid.fill(None)

    # 3. Inject our trained elites
    for agent in saved_agents:
        if getattr(agent, 'faction', '') == 'hunter':
            agent.health = 40.0
        else:
            agent.health = 100.0
        agent.age = 0
        world.add_agent(agent)

    # We overwrite the evolution method to rewind time instead of breeding
    def loop_simulation():
        print("\n--- Epoch Complete. Rewinding Time ---")
        
        # 1. Reset the environment (resets tick to 0, storm, scent, and island resources)
        world._init_epoch()
        
        # 2. Clear out the current agents on the board and in the graveyard
        world.agents.clear()
        world.agent_grid.fill(None)
        if hasattr(world, 'graveyard'):
            world.graveyard.clear()

        # 3. Reload agents from the pickle file
        with open(checkpoint_path, "rb") as f:
            fresh_agents = pickle.load(f)

        # 4. Inject them back for the next loop
        for agent in fresh_agents:
            if getattr(agent, 'faction', '') == 'hunter':
                agent.health = 40.0
            else:
                agent.health = 100.0
            agent.age = 0
            world.add_agent(agent)

    # Monkey-patch the evaluation method
    world.evaluate_n_evolve = loop_simulation

    return world


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay a trained checkpoint")
    parser.add_argument(
        "--file", 
        type=str, 
        help="Path to the checkpoint pickle file. Defaults to latest."
    )
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "3", "4"],
        help="World type: 1=Random, 2=Smart, 3=Two Island, 4=Competitive",
    )
    args = parser.parse_args()

    scenario_name = None
    world_fn = None
    if args.scenario:
        scenario_name, world_fn = SCENARIOS[args.scenario]

    target_file = args.file if args.file else get_latest_checkpoint(scenario_name)

    if not target_file or not os.path.exists(target_file):
        search_dir = os.path.join(CHECKPOINT_DIR, scenario_name) if scenario_name else CHECKPOINT_DIR
        print(f"Could not find any checkpoints in '{search_dir}/'.")
        print("Make sure you run headless_runner.py first!")
        sys.exit(1)

    # Boot up your existing visualizer using our custom creation function!
    # We wrap it in a lambda because load_checkpoint_world requires an argument.
    run_visualization(
        world_creation_fun=lambda: load_checkpoint_world(target_file, world_fn),
        show_scent_heatmap=True,
        show_scent_vectors=False    
    )


if __name__ == "__main__":
    main()