import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from brain import Brain
from world import World
from agent import Agent, InputSchema, Action

WIDTH = 50
HEIGTH = 50
INITIAL_AGENTS = 50
STEPS_PER_FRAME = 1 # i.e. simulation steps per animation frame
INIT_HEALTH = 100.0
INIT_RES_DENSITY = 0.2
HIDDEN_DIM = 10
OUTPUT_DIM = len(Action)
INPUT_DIM = InputSchema.TOTAL_INPUTS

def create_world():
    world = World(WIDTH, HEIGTH, init_res_density=INIT_RES_DENSITY)

    for _ in range(INITIAL_AGENTS):
        world.add_agent(
            Agent(
                Brain(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM), 
                np.random.randint(0, world.width), 
                np.random.randint(0, world.heigth), 
                INIT_HEALTH, 
                tuple(np.random.rand(3))
            )
        )

    return world

# An LLM was used to help me with the visualization
def run_visualization():
    world = create_world()

    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_title("Co-evolution Grid-World")

    # initialize the resource reference for the plot, will be updated subsequently
    resouce_ref = ax.imshow(world.resource_grid.T, cmap="Greens", vmin=0, vmax=5, origin="lower")

    # same for the agents reference
    agents_img = ax.scatter([], [], c=[], s=50, edgecolors="black")

    stats_text_ref = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    def update(frame):
        # Run simulation
        for _ in range(STEPS_PER_FRAME):
            world.update_world()

        # Now let's update stuff
        resouce_ref.set_data(world.resource_grid.T)

        if world.agents:
            x_s = [a.x for a in world.agents]
            y_s = [a.y for a in world.agents]
            colors = [a.color for a in world.agents]

            # update scatter plot
            agents_img.set_offsets(np.c_[x_s, y_s])
            agents_img.set_color(colors)        
        else:
            # clear if dead
            agents_img.set_offsets(np.empty((0, 2)))

        # Update stats_text_ref
        avg_health = np.mean([a.health for a in world.agents]) if world.agents else 0
        stats_text_ref.set_text(f"Generation: {frame}\nPopulation: {len(world.agents)}\nAverage Health: {avg_health:.2f}")

        return resouce_ref, agents_img, stats_text_ref

    anim = animation.FuncAnimation(fig, update, interval=50, blit=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualization()
