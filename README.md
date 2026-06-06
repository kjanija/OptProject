A grid-based artificial life simulation built from scratch.

- Neural Network "Brains": agents are driven by feed forward neural networks. Weights are mutated and passed down to offspring
- Phenotype Visualization: an agent's RGB color is dynamically generated based on its genes. So for example red indicates aggression, green foraging and blue evasive bahaviours.

## Requirements

```python
pip install numpy matplotlib tqdm
```

## Usage

### 1. Showcase

To simply run one of the predefined simulation scenarios, run:

```python
python scripts/run_showcase.py
```

### 2. Headless

Use this script to run headless simulations for statistical analysis. GUI is disabled, so it is faster. For now it saves action frequencies and population sizes in a CSV file. Saving entire trajectories is under consideration

```python
python scripts/run_headless.py
```

### 3. Default Random Simulation

Runs the simulation with a population of randomly initialized agents in a standard world.

```python
python -m optproject.runners.visualization_runner
```

Generational demo:

```python
python scripts/run_generational.py
```

## Project Structure

- Core logic lives under src/optproject/core/ (actions, schema, agent, brain, world_base).
- Environments in src/optproject/environments/ (classic, oasis, generational worlds).
- Scenario builders in src/optproject/scenarios/ for showcase and generational runs.
- Runners in src/optproject/runners/ for visualization and headless execution.
- scripts/ contains CLI entrypoints mirroring the original scripts.
