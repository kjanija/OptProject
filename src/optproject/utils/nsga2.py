from platypus import Problem, Solution, nondominated_sort, crowding_distance

from optproject.core.agent import Agent

def get_nsga2_elites(agents: list[Agent], num_elites: int) -> list[Agent]:
    """
    Uses platypus to perform non dominated sorting and crowding distance 
    calculation to select elites
    """

    num_elites = min(num_elites, len(agents))

    problem = Problem(1, 3)  # 1 decision variable (agent), 3 objectives (distance, health, age)
    
    solutions = []

    # map custom agents to platypus solutions
    for agent in agents:
        sol = Solution(problem)
        # we negate because platypus minimizes by default
        sol.objectives[0] = -agent.norm_x  # type: ignore
        sol.objectives[1] = -agent.norm_h   # type: ignore
        sol.objectives[2] = -agent.norm_age # type: ignore
        sol.agent = agent
        solutions.append(sol)

    # pareto fronts
    nondominated_sort(solutions)

    solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))  # sort by front rank, then crowding distance

    # elitism
    elites = [sol.agent for sol in solutions[:num_elites]]

    return elites

def get_twoisland_elites(agents: list[Agent], num_elites: int) -> list[Agent]:
    """
    3 Objectives: Proximity to Island 1, Proximity to Island 2, and Health.
    """
    num_elites = min(num_elites, len(agents))
    problem = Problem(1, 3) 
    solutions = []

    for agent in agents:
        sol = Solution(problem)
        # We negate because platypus minimizes by default
        sol.objectives[0] = -agent.norm_prox_1 # type: ignore
        sol.objectives[1] = -agent.norm_prox_2 # type: ignore
        sol.objectives[2] = -agent.norm_h      # type: ignore
        sol.agent = agent
        solutions.append(sol)

    nondominated_sort(solutions)
    solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))
    
    return [sol.agent for sol in solutions[:num_elites]]

def get_migrator_elites(agents: list[Agent], num_elites: int) -> list[Agent]:
    """
    3 Objectives: Distance, Health, Age.
    """
    num_elites = min(num_elites, len(agents))
    problem = Problem(1, 3) 
    solutions = []

    for agent in agents:
        sol = Solution(problem)
        sol.objectives[0] = -agent.norm_x  # type: ignore
        sol.objectives[1] = -agent.norm_h   # type: ignore
        sol.objectives[2] = -agent.norm_age # type: ignore
        sol.agent = agent
        solutions.append(sol)

    nondominated_sort(solutions)
    solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))
    
    return [sol.agent for sol in solutions[:num_elites]]


def get_hunter_elites(agents: list[Agent], num_elites: int) -> list[Agent]:
    """
    2 Objectives: Health, Age (Distance is ignored so they don't just run away).
    """
    num_elites = min(num_elites, len(agents))
    problem = Problem(1, 2) 
    solutions = []

    for agent in agents:
        sol = Solution(problem)
        sol.objectives[0] = -agent.norm_h   # type: ignore
        sol.objectives[1] = -agent.norm_age # type: ignore
        sol.agent = agent
        solutions.append(sol)

    nondominated_sort(solutions)
    solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))
    
    return [sol.agent for sol in solutions[:num_elites]]