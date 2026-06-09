import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_experiment_data(csv_file="experiment_data.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}. Make sure you are running this from the correct directory.")
        sys.exit(1)

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Calculate dynamic marker spacing (approx 20 markers total across the line)
    marker_spacing = max(1, len(df) // 20)

    # Set up a 4-panel figure now
    fig, (ax_dist, ax_surv, ax_move, ax_rest) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Co-Evolutionary Training Dashboard", fontsize=18, fontweight="bold", y=0.98)

    # ---------------------------------------------------------
    # Panel 1: Scenario-specific Objective
    # ---------------------------------------------------------
    if "Max_Dist" in df.columns:
        ax_dist.plot(df["Generation"], df["Max_Dist"], label="Max Distance (Elites)", color="#d62728", linewidth=2)
        ax_dist.plot(df["Generation"], df["Avg_Dist"], label="Avg Distance (Population)", color="#ff9896", linewidth=2)
        ax_dist.set_title("Distance to Island", fontsize=14)
        ax_dist.set_ylabel("Normalized Distance")
    elif "Max_Prox_1" in df.columns:
        ax_dist.plot(df["Generation"], df["Max_Prox_1"], label="Max Prox 1", color="#d62728", linewidth=2)
        ax_dist.plot(df["Generation"], df["Max_Prox_2"], label="Max Prox 2", color="#1f77b4", linewidth=2)
        ax_dist.plot(df["Generation"], df["Avg_Prox_1"], label="Avg Prox 1", color="#ff9896", linewidth=2, linestyle="--")
        ax_dist.plot(df["Generation"], df["Avg_Prox_2"], label="Avg Prox 2", color="#aec7e8", linewidth=2, linestyle="--")
        ax_dist.set_title("Proximity to Islands", fontsize=14)
        ax_dist.set_ylabel("Normalized Proximity")
    elif "Max_Dist_Migrator" in df.columns:
        ax_dist.plot(df["Generation"], df["Max_Dist_Migrator"], label="Max Dist (Migrator)", color="#d62728", linewidth=2)
        ax_dist.plot(df["Generation"], df["Avg_Dist_Migrator"], label="Avg Dist (Migrator)", color="#ff9896", linewidth=2)
        ax_dist.set_title("Distance to Island (Migrators)", fontsize=14)
        ax_dist.set_ylabel("Normalized Distance")
        
    ax_dist.set_ylim(0, 1.05)
    ax_dist.grid(True, linestyle="--", alpha=0.6)
    ax_dist.legend(loc="upper left")

    # ---------------------------------------------------------
    # Panel 2: Survivability (Objectives 2 & 3)
    # ---------------------------------------------------------
    if "Avg_Health_Migrator" in df.columns:
        color_health_m = "#2ca02c"
        color_health_h = "#d62728"
        ax_surv.plot(df["Generation"], df["Avg_Health_Migrator"], label="Avg Health (Migrator)", color=color_health_m, linewidth=2)
        ax_surv.plot(df["Generation"], df["Avg_Health_Hunter"], label="Avg Health (Hunter)", color=color_health_h, linewidth=2)
        ax_surv.set_ylabel("Avg Health", color="#000000", fontweight="bold")
        ax_surv.tick_params(axis='y', labelcolor="#000000")
        ax_surv.set_ylim(0, 1.05)
    else:
        color_health = "#2ca02c"
        ax_surv.plot(df["Generation"], df["Avg_Health"], label="Avg Health", color=color_health, linewidth=2)
        ax_surv.set_ylabel("Avg Health", color=color_health, fontweight="bold")
        ax_surv.tick_params(axis='y', labelcolor=color_health)
        ax_surv.set_ylim(0, 1.05)

    ax_age = ax_surv.twinx()
    color_age = "#1f77b4"
    ax_age.plot(df["Generation"], df["Avg_Age"], label="Avg Age", color=color_age, linewidth=2, linestyle="--")
    ax_age.set_ylabel("Avg Age (Ticks)", color=color_age, fontweight="bold")
    ax_age.tick_params(axis='y', labelcolor=color_age)
    ax_age.set_ylim(0, 210)

    ax_surv.set_title("Survivability (Health vs. Lifespan)", fontsize=14)
    ax_surv.grid(True, linestyle="--", alpha=0.6)

    lines_1, labels_1 = ax_surv.get_legend_handles_labels()
    lines_2, labels_2 = ax_age.get_legend_handles_labels()
    ax_surv.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    # ---------------------------------------------------------
    # Action Encodings (Colors + Markers)
    # ---------------------------------------------------------
    okabe_ito_dict = {
        "GATHER": "#009E73",               # Bluish Green
        "GIVE": "#56B4E9",                 # Sky Blue
        "TAKE": "#D55E00",                 # Vermilion
        "WAIT": "#000000",                 # Black
        "MOVE_TO_RESOURCE": "#0072B2",     # Blue
        "MOVE_AWAY_FROM_AGENT": "#E69F00", # Orange
        "MOVE_TOWARDS_AGENT": "#CC79A7",   # Reddish Purple
        "MOVE_RANDOM": "#999999",          # Grey
        "MOVE_TO_SCENT": "#F0E422",        # Yellow
    }

    action_markers = {
        "GATHER": "o",               # Circle
        "GIVE": "s",                 # Square
        "TAKE": "^",                 # Triangle Up
        "WAIT": "x",                 # Cross
        "MOVE_TO_RESOURCE": "D",     # Diamond
        "MOVE_AWAY_FROM_AGENT": "v", # Triangle Down
        "MOVE_TOWARDS_AGENT": "p",   # Pentagon
        "MOVE_RANDOM": "*",          # Star
        "MOVE_TO_SCENT": "h",        # Hexagon
    }

    action_cols = [col for col in df.columns if col.startswith("Action_")]
    
    # Split the actions into two groups
    move_cols = [col for col in action_cols if "MOVE" in col]
    rest_cols = [col for col in action_cols if "MOVE" not in col]

    # Dictionary to keep track of lines for the interactive legend
    interactive_lines = {}

    def plot_action_group(ax, columns, title, is_bottom=False):
        for col in columns:
            label = col.replace("Action_", "")
            color = okabe_ito_dict.get(label, "#999999")
            marker = action_markers.get(label, "")

            (line,) = ax.plot(
                df["Generation"], 
                df[col], 
                label=label, 
                color=color, 
                marker=marker,
                markersize=7,
                markevery=marker_spacing,
                linewidth=2,
                alpha=0.85
            )
            interactive_lines[label] = line

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Total Count")
        ax.grid(True, linestyle="--", alpha=0.6)
        if is_bottom:
            ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
        
        # External Legend
        return ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="Actions")

    # ---------------------------------------------------------
    # Panel 3: Move Actions
    # ---------------------------------------------------------
    leg_move = plot_action_group(ax_move, move_cols, "Navigation Strategy (Move Actions)")

    # ---------------------------------------------------------
    # Panel 4: Rest Actions (Interactions)
    # ---------------------------------------------------------
    leg_rest = plot_action_group(ax_rest, rest_cols, "Interaction Strategy (Non-Move Actions)", is_bottom=True)

    # ---------------------------------------------------------
    # Interactive Legend Logic (Applies to both action panels)
    # ---------------------------------------------------------
    lined = {}
    
    # Map the legend lines to the actual plot lines for both legends
    for leg in [leg_move, leg_rest]:
        for legline, label_text in zip(leg.get_lines(), [t.get_text() for t in leg.get_texts()]):
            legline.set_picker(True)
            legline.set_pickradius(5)
            # Find the actual data line associated with this legend text
            lined[legline] = interactive_lines[label_text]

    def on_pick(event):
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------
    plt.tight_layout()
    # Adjust right margin so both external legends fit nicely
    plt.subplots_adjust(right=0.82) 
    plt.show()

if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "experiment_data.csv"
    plot_experiment_data(target_csv)