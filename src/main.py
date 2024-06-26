from typing import *

from action_spaces import HOActionSpaceA
from state_spaces import HOStateA, HOStateB
from learners import ModelType
from sweep import Sweep
from plot import replot_figures


if __name__ == "__main__":
    Sweep(
        model_name="gpt-4",
        temperature=0.9,
        prompt_name="msamogh/sl-2",
        dataset_name="persistsim-sweep-15",
        dataset_split="valid",
        persistence_levels=[1, 5, 10],
        geometry_proficiency_levels=[3],
        model_type=[
            # ModelType.THEORETICAL,
            # ModelType.THEORY_PLUS_COMPUTATIONAL,
            ModelType.THEOR_COMP_BEHAV,
        ],
        states=[HOStateB()],
        action_space=HOActionSpaceA(),
    ).run(create_dataset=True)
    replot_figures()
