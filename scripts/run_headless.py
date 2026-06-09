from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.runners.headless_runner import run_headless
from optproject.runners.headless_generational import run_headless as run_headless_generational
from optproject.scenarios.generational_scenarios import (
    create_competitive_world,
    create_random_escape_world,
    create_smart_escape_world,
    create_two_island_world,
)


SCENARIOS = {
    1: ("random", create_random_escape_world),
    2: ("smart", create_smart_escape_world),
    3: ("two-island", create_two_island_world),
    4: ("competitive", create_competitive_world),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run headless simulation")
    parser.add_argument(
        "--mode",
        choices=("headless", "generational"),
        default="generational",
        help="Choose the headless runner to execute",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=tuple(SCENARIOS.keys()),
        default=1,
        help="Scenario to use for the generational runner: 1=random, 2=smart, 3=two-island, 4=competitive",
    )
    args = parser.parse_args()
    if args.mode == "generational":
        _, world_creation_fun = SCENARIOS[args.scenario]
        run_headless_generational(world_creation_fun=world_creation_fun)
    else:
        run_headless()


if __name__ == "__main__":
    main()
