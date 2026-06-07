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

# Ensure these match the constants used during training
WIDTH = 50
HEIGTH = 50
INIT_RES_DENSITY = 0.2
CHECKPOINT_DIR = "checkpoints"


def get_latest_checkpoint():
    """Finds the most recently modified pickle file."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pkl"))
    if not files:
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_checkpoint_world(checkpoint_path):
    """
    This acts as our custom 'world_creation_fun'. 
    It loads the agents, builds the world, and returns it.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        saved_agents = pickle.load(f)

    print(f"Loaded {len(saved_agents)} trained agents. Initializing world...")

    # 1. Create a fresh Generational World
    world = GenerationalWorld(width=WIDTH, height=HEIGTH, init_res_density=INIT_RES_DENSITY)
    
    # 2. Clear out whatever random agents it spawned by default
    world.agents = []
    world.agent_grid.fill(None)

    # 3. Inject our trained elites
    for agent in saved_agents:
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
            agent.health = 100.0
            agent.age = 0
            world.add_agent(agent)

    # Monkey-patch the evaluation method
    world.evaluate_n_evolve = loop_simulation

    return world


def main():
    # Allow the user to specify a file, or default to the newest one
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = get_latest_checkpoint()

    if not target_file or not os.path.exists(target_file):
        print(f"Could not find any checkpoints in '{CHECKPOINT_DIR}/'.")
        print("Make sure you run headless_runner.py first!")
        sys.exit(1)

    # Boot up your existing visualizer using our custom creation function!
    # We wrap it in a lambda because load_checkpoint_world requires an argument.
    run_visualization(
        world_creation_fun=lambda: load_checkpoint_world(target_file),
        show_scent_heatmap=True,
        show_scent_vectors=False    
    )


if __name__ == "__main__":
    main()