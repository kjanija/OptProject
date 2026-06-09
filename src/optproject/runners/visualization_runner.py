import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..core.actions import Action
from ..core.agent import Agent
from ..core.brain import Brain
from ..core.schema import InputSchema
from ..core.world_base import World
from ..core.config import (
    WIDTH, HEIGTH, INIT_AGENTS, INIT_HEALTH, 
    INIT_RES_DENSITY, HIDDEN_DIM, STEPS_PER_FRAME
)

OUTPUT_DIM = len(Action)
INPUT_DIM = InputSchema.TOTAL_INPUTS


def create_world():
    world = World(WIDTH, HEIGTH, init_res_density=INIT_RES_DENSITY)

    popsize = 0
    while popsize < INIT_AGENTS:
        brain = Brain(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        x, y = np.random.randint(0, world.width), np.random.randint(0, world.heigth)
        agent = Agent(brain, x, y, INIT_HEALTH, color=None)
        if world.add_agent(agent):
            popsize += 1

    return world


# An LLM was used to help me with the visualization


def run_visualization(
    world_creation_fun=create_world,
    show_scent_heatmap=True,
    show_scent_vectors=False,
    vector_step=4,
    scent_heatmap_vmin: float = 1e-2,
    scent_heatmap_alpha: float = 0.55,
    scent_vector_gain: float = 25.0,
    scent_vector_use_log: bool = True,
):
    world = world_creation_fun()
    animation_handle = {"ani": None}

    scent_data = world.scent_grid.T
    scent_vmax = max(float(np.max(scent_data)), scent_heatmap_vmin * 10.0)
    scent_norm = LogNorm(vmin=scent_heatmap_vmin, vmax=scent_vmax)

    # Create figure with 2 subplots (Grid on top, Stats on bottom)
    fig, (ax_grid, ax_stats) = plt.subplots(
        2, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_grid.set_title("Co-evolution Grid World")
    ax_stats.set_title("Action Distribution (Last Step)")
    ax_stats.set_ylabel("Count")
    ax_stats.set_xlabel("Step")

    # 1. Resource Grid (Image)
    resource_img = ax_grid.imshow(
        np.ma.masked_less_equal(world.resource_grid.T, 0.0), 
        cmap="Greens", 
        vmin=0, 
        vmax=5, 
        origin="lower", 
        zorder=3
    )

    # Optional: Scent Heatmap Overlay
    scent_img = None
    if show_scent_heatmap and hasattr(world, "scent_grid"):
        scent_img = ax_grid.imshow(
            np.ma.masked_less_equal(world.scent_grid.T, 0.0),
            cmap="Purples",
            norm=scent_norm,
            origin="lower",
            alpha=scent_heatmap_alpha,
            zorder=2,
        )

    # Optional: Scent Vector Field (Quiver Plot)
    quiver = None
    if show_scent_vectors and hasattr(world, "scent_grid"):
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(world.width), np.arange(world.heigth))

        # Calculate mathematical gradient (derivative) of the scent
        scent_field = (
            np.log1p(world.scent_grid.T) if scent_vector_use_log else world.scent_grid.T
        )
        dy, dx = np.gradient(scent_field)

        # Slice to subsample the arrows and reduce screen clutter
        skip = (slice(None, None, vector_step), slice(None, None, vector_step))

        quiver = ax_grid.quiver(
            x_grid[skip],
            y_grid[skip],
            dx[skip] * scent_vector_gain,
            dy[skip] * scent_vector_gain,
            color="purple",
            alpha=0.8,
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=1,
            zorder=4,
        )

    # 2. Agents (Scatter Plot)
    agents_scatter = ax_grid.scatter([], [], c=[], s=50, edgecolors="black", zorder=5)

    # Storm Line
    storm_line = None
    if hasattr(world, "storm_x"):
        storm_line = ax_grid.axvline(x=world.storm_x, color="red", linestyle="--", linewidth=2, zorder=6, alpha=0.8)

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

    # Define markers to pair with the colors (Double Encoding)
    action_markers = {
        Action.GATHER: "o",               # Circle
        Action.GIVE: "s",                 # Square
        Action.TAKE: "^",                 # Triangle Up
        Action.WAIT: "x",                 # Cross
        Action.MOVE_TO_RESOURCE: "D",     # Diamond
        Action.MOVE_AWAY_FROM_AGENT: "v", # Triangle Down
        Action.MOVE_TOWARDS_AGENT: "p",   # Pentagon
        Action.MOVE_RANDOM: "*",          # Star
        Action.MOVE_TO_SCENT: "h",        # Hexagon
    }

    for action in Action:
        (line,) = ax_stats.plot(
            [],
            [],
            label=action.name,
            color=action_colors.get(action, "black"),
            linestyle=action_styles.get(action, "-"),
            marker=action_markers.get(action, ""),
            markersize=6,
            markevery=5,  # Only draw a marker every 5 steps so it doesn't get cluttered
            linewidth=2,
        ) 
        lines[action] = line

    # Generate the legend
    leg = ax_stats.legend(loc="upper right", fontsize="small", ncol=2)

    # --- MAKE LEGEND INTERACTIVE ---
    # Map the legend lines to the actual plot lines
    lined = {}
    for legline, origline in zip(leg.get_lines(), lines.values()):
        legline.set_picker(True)  # Enable clicking on the legend line
        legline.set_pickradius(5) # Make it easier to click
        lined[legline] = origline

    def on_pick(event):
        # On the click event, find which legend item was clicked
        legline = event.artist
        origline = lined[legline]
        
        # Toggle the visibility of the actual plot line
        visible = not origline.get_visible()
        origline.set_visible(visible)
        
        # Dim the legend item to show it is muted
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    # Connect the click event to the figure
    fig.canvas.mpl_connect('pick_event', on_pick)

    def update(frame):
        # Run Simulation
        for _ in range(STEPS_PER_FRAME):
            world.update_world()
            if hasattr(world, "evaluate_n_evolve"):
                if world.tick >= getattr(world, "max_ticks", float("inf")) or not world.agents:
                    world.evaluate_n_evolve()

        if hasattr(world, "generation"):
            step_label = f"Gen: {getattr(world, 'generation', 0)} | Step: {getattr(world, 'tick', frame)}"
        else:
            step_label = f"Step: {frame}"

        if not world.agents:
            stats_text.set_text(f"{step_label}\nPop: 0\nExtinct")
            ani = animation_handle["ani"]
            if ani is not None:
                ani.event_source.stop()
            return resource_img, agents_scatter, stats_text, *lines.values()

        # Update Grid Visualization
        resource_img.set_data(np.ma.masked_less_equal(world.resource_grid.T, 0.0))

        # Update Scent Visualization
        if hasattr(world, "scent_grid"):
            if scent_img is not None:
                scent_vmax = max(float(np.max(world.scent_grid)), scent_heatmap_vmin * 10.0)
                scent_img.set_norm(LogNorm(vmin=scent_heatmap_vmin, vmax=scent_vmax))
                scent_img.set_data(world.scent_grid.T)

            if quiver is not None:
                scent_field = (
                    np.log1p(world.scent_grid.T)
                    if scent_vector_use_log
                    else world.scent_grid.T
                )
                dy, dx = np.gradient(scent_field)
                skip = (slice(None, None, vector_step), slice(None, None, vector_step))
                quiver.set_UVC(dx[skip] * scent_vector_gain, dy[skip] * scent_vector_gain)

        if world.agents:
            xs = [a.x for a in world.agents]
            ys = [a.y for a in world.agents]
            colors = [a.color for a in world.agents]
            agents_scatter.set_offsets(np.c_[xs, ys])
            agents_scatter.set_color(colors)
        else:
            agents_scatter.set_offsets(np.empty((0, 2)))

        # Update Storm Line
        if storm_line is not None and hasattr(world, "storm_x"):
            storm_line.set_xdata([world.storm_x, world.storm_x])

        avg_health = np.mean([a.health for a in world.agents]) if world.agents else 0
        stats_text.set_text(f"{step_label}\nPop: {len(world.agents)}\nHealth: {avg_health:.1f}")

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
    animation_handle["ani"] = ani # type: ignore
    plt.tight_layout()
    plt.show()
    return ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap", action="store_true", help="Show scent heatmap")
    parser.add_argument("--vectors", action="store_true", help="Show scent vectors")
    parser.add_argument("--vector-step", type=int, default=4, help="Vector sampling step")
    args = parser.parse_args()

    run_visualization(
        show_scent_heatmap=args.heatmap,
        show_scent_vectors=args.vectors,
        vector_step=args.vector_step,
    )


if __name__ == "__main__":
    main()
