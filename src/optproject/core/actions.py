from enum import IntEnum


class Action(IntEnum):
    """
    List of actions available to the agent.

    If you want to modify them, modify it here, since the agent's brain will resize
    according to the number of listed actions here
    """

    GATHER = 0  # Gather resource at current pos
    GIVE = 1  # Give health to neighbour
    TAKE = 2  # Steal health to neighbour
    WAIT = 3  # Do nothing
    MOVE_TO_RESOURCE = 4  # Move to neighbouring cell with most energy
    MOVE_AWAY_FROM_AGENT = 5  # Move away from neighbouring agents
    MOVE_TOWARDS_AGENT = 6  # Move towards neighbouring agents
    MOVE_RANDOM = 7  # Random movement
    MOVE_TO_SCENT = 8
