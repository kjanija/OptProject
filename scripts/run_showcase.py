from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.runners.visualization_runner import run_visualization # noqa: E402
from optproject.scenarios.showcase_scenarios import ( # noqa: E402
    scenario_altruist_parasite,
    scenario_boomBust_apocalypse,
    scenario_boomBust_oasis,
    scenario_boomBust_petriDish,
    scenario_cooldown,
    scenario_maturity,
    scenario_predator_prey,
    scenario_probabilistic,
    scenario_threshold,
)


def main():
    print("=========================================")
    print("   REPRODUCTION MECHANICS SHOWCASE       ")
    print("=========================================")
    print("0A: The Petri Dish (Full grid, 0 Regrowth)")
    print("0B: The Oasis (Center patch, Normal Regrowth)")
    print("0C: The Apocalypse (Center patch, 0 Regrowth)")
    print(" 1: Maturity Age (Wait to grow up)")
    print(" 2: Cooldown Time (Wait between kids)")
    print(" 3: Probabilistic (Chance to reproduce)")
    print(" 4: High Threshold (Must be wealthy)")
    print("=========================================")
    print(" 5: Predator vs Prey")
    print(" 6: Altruist and Parasite")

    choice = input("Enter selection (0A, 0B, 0C, 1, 2, 3, 4, 5, 6): ").strip().upper()

    scenarios = {
        "0A": scenario_boomBust_petriDish,
        "0B": scenario_boomBust_oasis,
        "0C": scenario_boomBust_apocalypse,
        "1": scenario_maturity,
        "2": scenario_cooldown,
        "3": scenario_probabilistic,
        "4": scenario_threshold,
        "5": scenario_predator_prey,
        "6": scenario_altruist_parasite,
    }

    if choice in scenarios:
        run_visualization(scenarios[choice])
    else:
        print("Invalid choice. Exiting.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
