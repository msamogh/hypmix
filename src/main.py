from typing import *

import randomname

from action_spaces import HOActionSpaceB
from state_spaces import HOStateB
from learners import ModelType
from sweep import Sweep
from plot import replot_figures
from langchain_utils import get_next_dataset_name


if __name__ == "__main__":
    DATASET_NAME = get_next_dataset_name()
    Sweep(
        sweep_id=randomname.get_name(),
        model_name="gpt-4",
        temperature=0.9,
        prompt_name="amogh-ld/sl-2",
        dataset_name="persistsim-sweep-39",
        dataset_split="valid",
        persistence_levels=[6],
        geometry_proficiency_levels=[2],
        model_types=[
            # ModelType.THEOR,
            # ModelType.THEOR_COMP,
            ModelType.THEOR_COMP_BEHAV,
        ],
        states=[
            HOStateB.generate_state(0, {}),
            HOStateB.generate_state(0, {"AF1": 1}),
            HOStateB.generate_state(0, {"AF1": 1, "AF2": 1}),
            HOStateB.generate_state(0, {"AF1": 1, "AF2": 1, "F1P": 1}),
            HOStateB.generate_state(
                0,
                {
                    "AP": 1,
                    "AF1": 1,
                    "AF2": 1,
                    "AO": 1,
                    "F1P": 1,
                    "F1F2": 1,
                    "F1O": 1,
                    "F2P": 1,
                    "F2O": 1,
                    "OP": 1,
                },
            ),
        ],
        action_space=HOActionSpaceB(),
    ).run()
    replot_figures()
