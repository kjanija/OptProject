from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.runners.headless_runner import run_headless
from optproject.runners.headless_generational import run_headless as run_headless_generational


def main():
    # cli argument to choose whether to run the generational or non-generational 
    # version of the headless runner, default is generational
    import argparse
    parser = argparse.ArgumentParser(description="Run headless simulation")
    parser.add_argument("--no-generational", action="store_true", help="Run the non-generational version of the headless runner")
    args = parser.parse_args()
    if not args.no_generational:
        run_headless_generational()
    else:
        run_headless()


if __name__ == "__main__":
    main()
