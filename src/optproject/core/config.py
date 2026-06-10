"""Global configuration constants for the OptProject simulation."""

# --- Grid & World Settings ---
WIDTH = 50
HEIGHT = 50
HEIGTH = 50  # Alias for historical misspelling
INIT_RES_DENSITY = 0.2
MAX_RESOURCE = 10.0
REGROW_DENSITY = 0.05
REGROW_AMOUNT = 5.0

# --- Epoch & Generational Settings ---
MAX_TICKS = 300
STORM_SPEED = 7  # Ticks per 1 unit of storm movement
ISLAND_WIDTH = 5  # Width of the resource island on the right
SPAWN_AREA_WIDTH = 4  # Width of the agent spawn area on the left

# --- Agent Core Settings ---
INIT_AGENTS = 100
INIT_HEALTH = 100.0
HIDDEN_DIM = 10
COST_OF_LIFE = 0.4
GIVE_AMOUNT = 10.0
TAKE_AMOUNT = 10.0
TAKE_COST_FRAC = 0.3

# --- Reproduction & Evolution Settings ---
REPRODUCTION_THRESHOLD = (
    75.0  # Health required to reproduce (in non-generational worlds)
)
REPRODUCTION_COST = 50.0
REPRODUCTION_MATURITY_AGE = 0
REPRODUCTION_COOLDOWN = 0
REPRODUCTION_PROB = 1.0
ELITES_FRACTION = 0.2
MUTATION_PROB = 0.1
MUTATION_AMP = 0.3

# --- Competitive Scenario Settings ---
MIGRATORS_FRACTION = 0.8
HUNTERS_HEALTH_MULTIPLIER = 0.4  # Hunters start with 40% of normal health
COST_OF_LIFE_COMPETITIVE = 0.6

# --- Scent Settings ---
SCENT_DECAY = 0.98
SCENT_DIFFUSION_RATE = 0.35
SCENT_SOURCE_STRENGTH = 100.0
SCENT_DIFFUSION_STEPS = 3
SCENT_INIT_DIFFUSION_STEPS = 150

# --- Training / Runner Settings ---
TOTAL_GENERATIONS = 2000
OUTPUT_FILE = "experiment_data.csv"

# --- Checkpointing Settings ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 100
MILESTONE_THRESHOLD = 0.95  # Save if an agent reaches N% of the distance

# --- Visualization Settings ---
STEPS_PER_FRAME = 1  # i.e. simulation steps per animation frame
