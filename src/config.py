import numpy as np

from environment.action_spaces import HOActionSpaceB, HOActionSpaceC, HOActionSpaceD
from environment.state_spaces import HOStateB

DATASET_NAME = "graph-1c"

ACTION_SPACE = HOActionSpaceC()
STATE_SWEEP_TINY = HOStateB.generate_uniform_state_space(size="tiny")
STATE_SWEEP_SMALL = HOStateB.generate_uniform_state_space(size="small")
STATE_SWEEP_MED = HOStateB.generate_uniform_state_space(size="medium")
STATE_SWEEP_UNIFORM_1 = HOStateB.generate_state_space_for_uniform_hyp()
STATE_SWEEP_THOROUGH_K = HOStateB.generate_thorough_state_space(ks=[3])
