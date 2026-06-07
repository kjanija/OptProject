import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.runners.visualization_runner import run_visualization # noqa: E402
from optproject.scenarios.generational_scenarios import ( # noqa: E402
    create_random_escape_world,
    create_smart_escape_world,
    create_two_island_world,
    create_competitive_world
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "3", "4"],
        default="1",
        help="1 = Blank Slate, 2 = Smart Injection, 3 = Two Island, 4 = Competitive",
    )
    parser.add_argument(
        "--show-scent-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show scent heatmap overlay",
    )
    parser.add_argument(
        "--show-scent-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show scent vector field overlay",
    )
    parser.add_argument(
        "--vector-step",
        type=int,
        default=4,
        help="Subsampling step for scent vectors",
    )
    parser.add_argument("--scent-decay", type=float, default=0.98)
    parser.add_argument("--scent-diffusion-rate", type=float, default=1)
    parser.add_argument("--scent-source-strength", type=float, default=100.0)
    parser.add_argument("--scent-diffusion-steps", type=int, default=3)
    parser.add_argument("--scent-init-diffusion-steps", type=int, default=150)

    args = parser.parse_args()

    scenario_map = {
        "1": (
            create_random_escape_world,
            "Starting... It may take 10+ generations before they learn to move Right!",
        ),
        "2": (
            create_smart_escape_world,
            "Starting... Watch them follow the pheromones immediately.",
        ),
        "3": (
            create_two_island_world,
            "Starting... They must navigate between two islands!",
        ),
        "4": (
            create_competitive_world,
            "Starting... Watch the hunters and migrators compete!",
        ),
    }

    create_world_fn, start_msg = scenario_map[args.scenario]

    def make_world():
        return create_world_fn(
            scent_decay=args.scent_decay,
            scent_diffusion_rate=args.scent_diffusion_rate,
            scent_source_strength=args.scent_source_strength,
            scent_diffusion_steps=args.scent_diffusion_steps,
            scent_init_diffusion_steps=args.scent_init_diffusion_steps,
        )

    print(start_msg)
    run_visualization(
        make_world,
        show_scent_heatmap=args.show_scent_heatmap,
        show_scent_vectors=args.show_scent_vectors,
        vector_step=args.vector_step,
    )


if __name__ == "__main__":
    main()
