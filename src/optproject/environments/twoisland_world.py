from .generational_world import GenerationalWorld

class TwoIslandWorld(GenerationalWorld):
    """
    Splits the survival objective into two physically distant locations 
    to test Diversity Maintenance (Speciation).
    """
    def _init_epoch(self):
        """Override to spawn TWO islands instead of one big one"""
        self.tick = 0
        self.storm_x = -1
        self.scent_grid.fill(0.0)
        self.resource_grid.fill(0.0)
        if hasattr(self, 'graveyard'):
            self.graveyard.clear()

        # Island 1 (Top Right)
        self.resource_grid[self.island_start_x :, 0 : self.heigth // 3] = self.max_resource
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = self.scent_source_strength
        
        # Island 2 (Bottom Right)
        self.resource_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.max_resource
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.scent_source_strength
        
        for _ in range(self.scent_init_diffusion_steps):
            self.diffuse_scent()

    def diffuse_scent(self):
        """Override to maintain the two distinct scent sources"""
        super().diffuse_scent()
        # Clear the default full-column scent applied by super()
        self.scent_grid[self.island_start_x :, :] = 0.0 
        
        # Re-apply the two distinct scent sources
        self.scent_grid[self.island_start_x :, 0 : self.heigth // 3] = self.scent_source_strength
        self.scent_grid[self.island_start_x :, 2 * self.heigth // 3 :] = self.scent_source_strength