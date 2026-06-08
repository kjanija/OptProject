# Co-Evolution & Game Theory in a Grid-World

A grid-based artificial life simulation built from scratch to explore **Co-Evolution and Game Theory**. 

- **Neural Network "Brains":** Agents are driven by feed-forward neural networks. Weights are mutated and passed down to offspring.
- **Phenotype Visualization:** An agent's RGB color is dynamically generated based on its genetic strategy. For example: Red indicates aggression (`TAKE`), Green indicates foraging/migration (`MOVE_TO_SCENT`, `GATHER`), and Blue indicates evasive/social behaviors (`MOVE_AWAY`, `GIVE`).
- **Multi-Objective Optimization:** Utilizes **NSGA-II** (via Platypus) to evaluate agents on conflicting objectives (e.g., Distance, Health, and Lifespan) while maintaining genetic diversity using Crowding Distance.

## Requirements

```bash
pip install numpy matplotlib tqdm pandas platypus-opt
```

## Usage

### 1. Generational Scenarios

Watch the agents evolve over epochs. I have implemented several scenarios to test different evolutionary pressures:


```bash
python scripts/run_generational.py --scenario <1|2|3|4>
```

where:

1. Blank Slate - Random mutants trying to survive a moving storm.
2. Smart Injection - Agents pre-loaded with scent-tracking genes.
3. Two-Island Dilemma - Tests speciation and diversity maintenance. Agents must choose between two physically separated islands.
4. Competitive Co-evolution - Blue Migrators try to reach the island while Red Hunters try to drain their health.

Note: You can toggle visual aids using --show-scent-heatmap and --show-scent-vectors.

### 2. Headless Training & Checkpointing

GUI visualization is slow. Use the headless runner to train.

```bash
python scripts/run_headless.py
```

This script will:

1. Run a 2000-generation training loop.
2. Save detailed population metrics (Max/Avg Distance, Health, Age, and specific Action Counts) to experiment_data.csv.
3. Save Pickle Checkpoints (.pkl files) of the agent population every 100 generations and at major evolutionary milestones inside a checkpoints/ directory.

### 3. The Training Dashboard

Analyze the results of the headless training run using the interactive dashboard.

```bash
python scripts/run_plots.py
```

This generates a 4-panel, colorblind-friendly (Okabe-Ito palette) Matplotlib dashboard showing distance progression, survivability curves, and action distributions. Click on the legend items to mute/unmute specific lines.

### 4. Replay

Load an exact checkpoint back into the GUI to watch the agents behave.

```bash
# Plays the most recent checkpoint
python scripts/run_replay.py 

# Or specify a specific generation
python scripts/run_replay.py checkpoints/gen_0100_INTERVAL.pkl
```

Note: The replay loop freezes the evolutionary cycle, seamlessly rewinding time at the end of the epoch so you can watch the agents repeat their strategies infinitely. --> it re-spawns the agents with the specific strategy they learned, but same actions are not guaranteed since there are non-deterministic components.

### 5. Game Theory Mechanics Showcase

To run continuous, non-generational simulations showcasing specific reproduction mechanics or game-theoretic archetypes (like Altruists vs. Parasites):

```bash
python scripts/run_showcase.py
```

## Project Structure

- `src/optproject/core/`: Core logic (`actions.py`, `agent.py`, `brain.py`, `world_base.py`, `schema.py`).
- `src/optproject/environments/`: Specialized world topologies (`generational_world.py`, `twoisland_world.py`, `competitive_world.py`).
- `src/optproject/scenarios/`: Configuration builders for specific experiments.
- `src/optproject/utils/`: Helper methods, at present only the NSGA-II Platypus integration (`nsga2.py`).
- `src/optproject/runners/`: Logic for visual and headless execution loops.
- `scripts/`: CLI entrypoints.
- `checkpoints/`: Auto-generated directory for .pkl milestone saves.