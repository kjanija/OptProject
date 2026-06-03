class InputSchema:
    """
    Indexes to identify the positions of various perceptions
    """

    ID_ENERGY = 0  # Energy levels in each cell in neighbouring 3x3 grid
    ID_OTHER_HEALTH = 9  # Health levels of agents in neighbouring 3x3 grid, 0 if empty
    ID_OTHER_DNA = 18  # Similarity to neighbouring agents, in (0.0, 1.0). Used to model "kinship"/"clans"
    ID_SELF_HEALTH = 27
    ID_SCENT = 28  # "scent values in 3x3 grid"

    TOTAL_INPUTS = 37
