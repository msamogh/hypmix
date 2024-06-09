from typing import *

from action_spaces import HOActionSpaceA
from state_spaces import HOStateA
from learners import PersistenceModel
from sweep import Sweep

from plot import replot_figures


if __name__ == "__main__":
    Sweep(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        prompt_name="msamogh/persistsim-trial",
        dataset_name="persistsim-sweep-2",
        dataset_split="calibration",
        persistence_level=[1, 2, 3],
        geometry_proficiency=[3],
        persistence_model=[
            PersistenceModel.PersistenceModelType.THEORETICAL,
            PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL,
            PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS,
        ],
        state=[HOStateA(time_elapsed_in_minutes=5, num_submission_attempts=0)],
        action_space=[HOActionSpaceA()],
    ).run()
    replot_figures()
