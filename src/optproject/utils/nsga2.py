from platypus import Problem, Solution, nondominated_sort, crowding_distance

def get_nsga2_elites(agents, num_elites):
    """
    Uses platypus to perform non dominated sorting and crowding distance 
    calculation to select elites
    """

    num_elites = min(num_elites, len(agents))

    problem = Problem(1, 2)
    
    solutions = []

    # map custom agents to platypus solutions
    for agent in agents:
        sol = Solution(problem)
        sol.objectives[0] = -agent.norm_x  # We negate because we want to maximize
        sol.objectives[1] = -agent.norm_h
        sol.agent = agent
        solutions.append(sol)

    # pareto fronts
    nondominated_sort(solutions)

    solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))  # sort by front rank, then crowding distance

    # elitism
    elites = [sol.agent for sol in solutions[:num_elites]]

    return elites