import numpy as np

from ..core.world_base import World


class OasisWorld(World):
    """
    Custom World with regrow only in the center
    """

    def resource_regrow(self, density: float = 0.01, amount: float = 5.0):
        if amount <= 0:
            return
        mask = np.random.rand(self.width, self.heigth) < density

        cx, cy = self.width // 2, self.heigth // 2
        # zero mask everywhere except middle patch
        mask[: cx - 5, :] = False
        mask[cx + 5 :, :] = False
        mask[:, : cy - 5] = False
        mask[:, cy + 5 :] = False

        self.resource_grid[mask] += amount
        self.resource_grid = np.minimum(self.resource_grid, self.max_resource)
