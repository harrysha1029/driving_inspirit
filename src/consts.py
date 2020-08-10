DIM = 96
CAR_ROW = 67
CAR_COL = 48

# =====================================

INDEX2ACTIONS = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (-1, 0, 0),
]

N_ACTIONS = len(INDEX2ACTIONS)

INDEX2STRING = [
    'stay',
    'accelerate',
    'right',
    'left',
]

# =====================================

ACTION2INDEX = {x: i for i, x in enumerate(INDEX2ACTIONS)}

# =====================================
