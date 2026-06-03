import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.core.actions import Action # noqa: E402
from optproject.core.agent import Agent # noqa: E402
from optproject.core.brain import Brain # noqa: E402


class AgentSmokeTests(unittest.TestCase):
    def test_agent_color(self):
        brain = Brain(37, 10, len(Action))
        agent = Agent(brain, 0, 0, 100.0, color=None)
        self.assertEqual(len(agent.color), 3)


if __name__ == "__main__":
    unittest.main()
