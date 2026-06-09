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

    args = parser.parse_args()
    create_world_fn, start_msg = scenario_map[args.scenario]

    print(start_msg)
    run_visualization(
        create_world_fn,
        show_scent_heatmap=args.show_scent_heatmap,
        show_scent_vectors=args.show_scent_vectors,
    )


if __name__ == "__main__":
    main()
