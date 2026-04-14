import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from agent import Action, Agent, InputSchema
from brain import Brain
from world import World

SHOW_SCENT_HEATMAP = True
SHOW_SCENT_VECTORS = False
VECTOR_STEP = 4

WIDTH = 50
HEIGTH = 50
INITIAL_AGENTS = 50
STEPS_PER_FRAME = 1  # i.e. simulation steps per animation frame
INIT_HEALTH = 100.0
INIT_RES_DENSITY = 0.2
HIDDEN_DIM = 10
OUTPUT_DIM = len(Action)
INPUT_DIM = InputSchema.TOTAL_INPUTS


def create_world():
    world = World(WIDTH, HEIGTH, init_res_density=INIT_RES_DENSITY)

    popsize = 0
    while popsize < INITIAL_AGENTS:
        brain = Brain(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        x, y = np.random.randint(0, world.width), np.random.randint(0, world.heigth)
        agent = Agent(brain,x, y, INIT_HEALTH, color=None)
        if world.add_agent(agent):
            popsize += 1

    return world


# An LLM was used to help me with the visualization


def run_visualization(world_creation_fun=create_world):
    world = world_creation_fun()

    # Create figure with 2 subplots (Grid on top, Stats on bottom)
    fig, (ax_grid, ax_stats) = plt.subplots(
        2, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_grid.set_title("Co-evolution Grid World")
    ax_stats.set_title("Action Distribution (Last Step)")
    ax_stats.set_ylabel("Count")
    ax_stats.set_xlabel("Generation")

    # 1. Resource Grid (Image)
    resource_img = ax_grid.imshow(
        world.resource_grid.T, cmap="Greens", vmin=0, vmax=5, origin="lower"
    )

    # Optional: Scent Heatmap Overlay
    scent_img = None
    if SHOW_SCENT_HEATMAP and hasattr(world, "scent_grid"):
        scent_img = ax_grid.imshow(
            world.scent_grid.T,
            cmap="Purples",
            vmin=0,
            vmax=100,
            origin="lower",
            alpha=0.4,
        )

    # Optional: Scent Vector Field (Quiver Plot)
    quiver = None
    if SHOW_SCENT_VECTORS and hasattr(world, "scent_grid"):
        # Create coordinate grids
        X, Y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGTH))

        # Calculate mathematical gradient (derivative) of the scent
        dy, dx = np.gradient(world.scent_grid.T)

        # Slice to subsample the arrows and reduce screen clutter
        skip = (slice(None, None, VECTOR_STEP), slice(None, None, VECTOR_STEP))

        quiver = ax_grid.quiver(
            X[skip], Y[skip], dx[skip], dy[skip], color="purple", alpha=0.7, pivot="mid"
        )

    # 2. Agents (Scatter Plot)
    agents_scatter = ax_grid.scatter([], [], c=[], s=50, edgecolors="black")

    # 3. Stats Text
    stats_text = ax_grid.text(
        0.02,
        0.98,
        "",
        transform=ax_grid.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 4. Action Line Chart Setup
    x_data = []
    # Create a list of lists to store history for each action
    y_data = {action: [] for action in Action}
    # Create lines for each action
    lines = {}

    # Define colorblind-friendly colors (Okabe-Ito palette)
    action_colors = {
        Action.GATHER: "#009E73",  # Bluish Green (Distinct from Vermilion)
        Action.GIVE: "#56B4E9",  # Sky Blue
        Action.TAKE: "#D55E00",  # Vermilion (Red-Orange)
        Action.WAIT: "#000000",  # Black
        Action.MOVE_TO_RESOURCE: "#0072B2",  # Blue
        Action.MOVE_AWAY_FROM_AGENT: "#E69F00",  # Orange
        Action.MOVE_TOWARDS_AGENT: "#CC79A7",  # Reddish Purple
        Action.MOVE_RANDOM: "#999999",  # Grey
        Action.MOVE_TO_SCENT: "#F0E422",  # yellow-ish
    }

    # Define linestyles to further distinguish categories (Double Coding)
    # Solid: Core Interaction
    # Dashed: Directed Movement
    # Dotted: Passive/Noise
    action_styles = {
        Action.GATHER: "-",
        Action.GIVE: "-",
        Action.TAKE: "-",
        Action.WAIT: ":",
        Action.MOVE_TO_RESOURCE: "--",
        Action.MOVE_AWAY_FROM_AGENT: "--",
        Action.MOVE_TOWARDS_AGENT: "--",
        Action.MOVE_RANDOM: ":",
        Action.MOVE_TO_SCENT: "--",
    }

    for action in Action:
        (line,) = ax_stats.plot(
            [],
            [],
            label=action.name,
            color=action_colors.get(action, "black"),
            linestyle=action_styles.get(action, "-"),
            linewidth=2,
        )  # Increased linewidth for better visibility
        lines[action] = line

    ax_stats.legend(loc="upper right", fontsize="small", ncol=2)

    def update(frame):
        # Run Simulation
        for _ in range(STEPS_PER_FRAME):
            world.update_world()

        # Update Grid Visualization
        resource_img.set_data(world.resource_grid.T)

        # Update Scent Visualization
        if hasattr(world, "scent_grid"):
            if scent_img is not None:
                scent_img.set_data(world.scent_grid.T)

            if quiver is not None:
                dy, dx = np.gradient(world.scent_grid.T)
                skip = (slice(None, None, VECTOR_STEP), slice(None, None, VECTOR_STEP))
                quiver.set_UVC(dx[skip], dy[skip])

        if world.agents:
            xs = [a.x for a in world.agents]
            ys = [a.y for a in world.agents]
            colors = [a.color for a in world.agents]
            agents_scatter.set_offsets(np.c_[xs, ys])
            agents_scatter.set_color(colors)
        else:
            agents_scatter.set_offsets(np.empty((0, 2)))

        avg_health = np.mean([a.health for a in world.agents]) if world.agents else 0
        stats_text.set_text(
            f"Gen: {frame}\nPop: {len(world.agents)}\nHealth: {avg_health:.1f}"
        )

        # Update Stats Chart
        x_data.append(frame)

        # Get stats from the world for this step
        current_stats = world.last_step_stats

        # Determine max y for scaling
        max_val = 0

        for action in Action:
            count = current_stats.get(action, 0)
            y_data[action].append(count)
            lines[action].set_data(x_data, y_data[action])
            if count > max_val:
                max_val = count

        # Dynamic scaling of axes
        ax_stats.set_xlim(
            max(0, frame - 100), frame + 10
        )  # Show window of last 100 frames
        ax_stats.set_ylim(0, max_val + 5)

        return resource_img, agents_scatter, stats_text, *lines.values()

    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=False, cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_visualization()
