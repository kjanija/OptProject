import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.core.actions import Action
from optproject.core.agent import Agent
from optproject.core.brain import Brain
from optproject.core.schema import InputSchema
from optproject.core.world_base import World


class WorldSmokeTests(unittest.TestCase):
    def test_world_creation(self):
        world = World(10, 10, init_res_density=0.0)
        self.assertEqual(world.width, 10)
        self.assertEqual(world.heigth, 10)

    def test_agent_addition(self):
        world = World(10, 10, init_res_density=0.0)
        brain = Brain(InputSchema.TOTAL_INPUTS, 10, len(Action))
        agent = Agent(brain, 1, 1, 100.0, color=None)
        self.assertTrue(world.add_agent(agent))


if __name__ == "__main__":
    unittest.main()
