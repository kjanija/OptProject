from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from optproject.core.actions import Action  # noqa: E402
from optproject.core.agent import Agent, run_agent_self_tests  # noqa: E402
from optproject.core.schema import InputSchema  # noqa: E402

__all__ = ["Action", "Agent", "InputSchema"]


if __name__ == "__main__":
    run_agent_self_tests()
