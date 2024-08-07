from environment.state_spaces import HOStateB
from environment.action_spaces import HOActionSpaceC, HOActionSpaceD


DATASET_NAME = "calibration-sprint-34"

ACTION_SPACE = HOActionSpaceD()
STATE_SWEEP_SMALL = HOStateB.generate_uniform_state_space(size="small")
STATE_SWEEP_MED = HOStateB.generate_uniform_state_space(size="medium")
