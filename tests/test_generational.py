import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from optproject.environments.generational_world import GenerationalWorld


class GenerationalWorldSmokeTests(unittest.TestCase):
    def test_generational_world_creation(self):
        world = GenerationalWorld(width=10, height=10)
        self.assertEqual(world.width, 10)
        self.assertEqual(world.heigth, 10)


if __name__ == "__main__":
    unittest.main()
