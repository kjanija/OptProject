from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from optproject.core.brain import Brain, BrainModel, run_brain_self_tests  # noqa: E402

__all__ = ["Brain", "BrainModel"]


if __name__ == "__main__":
    run_brain_self_tests()

