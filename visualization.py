from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from optproject.runners.visualization_runner import (  # noqa: E402
    create_world,
    main,
    run_visualization,
)

__all__ = ["create_world", "run_visualization"]


if __name__ == "__main__":
    main()
