from environment.state_spaces import HOStateB
from environment.action_spaces import HOActionSpaceB, HOActionSpaceC, HOActionSpaceD


DATASET_NAME = "graph-1c"

ACTION_SPACE = HOActionSpaceC()
STATE_SWEEP_SMALL = HOStateB.generate_uniform_state_space(size="small")
STATE_SWEEP_MED = HOStateB.generate_uniform_state_space(size="medium")
