from environment.state_spaces import HOStateB
from environment.action_spaces import HOActionSpaceB


DATASET_NAME = "calibration-sprint-1"

ACTION_SPACE = HOActionSpaceB()
STATE_SWEEP = HOStateB.generate_uniform_state_space()

