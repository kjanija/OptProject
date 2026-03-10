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
python showcase.py
```

### 2. Headless

Use this script to run headless simulations for statistical analysis. GUI is disabled, so it is faster. For now it saves action frequencies and population sizes in a CSV file. Saving entire trajectories is under consideration

```python
python headless.py
```

### 3. Default Random Simulation

Runs the simulation with a population of randomly initialized agents in a standard world.

```python
python visualization.py
```

## Project Structure

- `world.py`: the environment engine. Handles 2D toroidal grid, resource regrowth and main step loop (aging, metabolism, reproduction, actions).
- `agent.py`: biological unit. Manages health, position, reproduction and the genetic color-mapping.
- `brain.py`: contains the `BrainModel` interface and the Brain (Feed forward NN) implementation. Handles `predict` and `clone_and_mutate`.
- `showcase.py`: hardcoded smart genes and custom world setup to explicitly demonstrate specific evolutionary mechanics.
