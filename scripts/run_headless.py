from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.runners.headless_runner import run_headless


def main():
    run_headless()


if __name__ == "__main__":
    main()
