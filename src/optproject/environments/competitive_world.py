from .generational_world import GenerationalWorld
import random
from ..utils.nsga2 import get_migrator_elites, get_hunter_elites

class CompetitiveWorld(GenerationalWorld):
    """
    Two-Population Competitive Coevolution.
    Migrators try to reach the island. Hunters try to drain their health.
    """
    def evaluate_n_evolve(self):
        pool = self.agents + getattr(self, 'graveyard', [])
        
        migrators = [a for a in pool if getattr(a, 'faction', '') == "migrator"]
        hunters = [a for a in pool if getattr(a, 'faction', '') == "hunter"]
        
        self.agents = []
        self.agent_grid.fill(None)
        if hasattr(self, 'graveyard'):
            self.graveyard.clear()
        
        max_x = float(self.width)
        max_ticks = float(self.max_ticks)
        
        # --- 1. Evolve Migrators (Distance + Health + Age) ---
        if not migrators:
            print(f"Gen {self.generation}: Migrator Extinction.")
        else:
            # Normalize Migrator stats
            max_h_m = max([a.health for a in migrators])
            max_h_m = max(1.0, max_h_m)
            for m in migrators:
                m.norm_x = m.x / max_x # type: ignore
                m.norm_h = max(0.0, m.health) / max_h_m # type: ignore
                m.norm_age = m.age / max_ticks # type: ignore
            
            # NSGA-II Platypus Evaluation
            num_migrator_elites = max(1, len(migrators) // 5) # Top 20%
            top_migrators = get_migrator_elites(migrators, num_migrator_elites)
            
            # Breed 40 Migrators
            while len([a for a in self.agents if getattr(a, 'faction', '') == "migrator"]) < 40:
                child = random.choice(top_migrators).reproduce(0.1, 0.2, 0.0)
                child.faction = "migrator" # type: ignore
                nx = random.randint(0, 3) 
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    child.x, child.y = nx, ny
                    child.health, child.age = 100.0, 0
                    self.add_agent(child)
        
        # --- 2. Evolve Hunters (Health + Age) ---
        if not hunters:
            print(f"Gen {self.generation}: Hunter Extinction.")
        else:
            # Normalize Hunter stats
            max_h_h = max([a.health for a in hunters])
            max_h_h = max(1.0, max_h_h)
            for h in hunters:
                # Note: No norm_x here!
                h.norm_h = max(0.0, h.health) / max_h_h # type: ignore
                h.norm_age = h.age / max_ticks # type: ignore
            
            # NSGA-II Platypus Evaluation
            num_hunter_elites = max(1, len(hunters) // 5) # Top 20%
            top_hunters = get_hunter_elites(hunters, num_hunter_elites)
            
            # Breed 10 Hunters
            while len([a for a in self.agents if getattr(a, 'faction', '') == "hunter"]) < 10:
                child = random.choice(top_hunters).reproduce(0.1, 0.2, 0.0)
                child.faction = "hunter" # type: ignore
                nx = random.randint(15, 20) # Spawn in the middle to intercept!
                ny = random.randint(0, self.heigth - 1)
                if self.agent_grid[nx, ny] is None:
                    child.x, child.y = nx, ny
                    child.health, child.age = 40.0, 0
                    self.add_agent(child)
                    
        self.generation += 1
        self._init_epoch()