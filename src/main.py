import random
from typing import *

import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import config


def main(dataset_name, action_space, tgt_hyp):
    from experiments.mdhyp import (
        MonotonicCalibratedB,
        MonotonicCalibratedE,
        MonotonicCalibratedI,
        MonotonicUncalibrated,
        UniformCalibratedF,
        UniformCalibratedH,
        UniformDistributionUncalibrated,
    )
    from learner.geometry_proficiency import (
        proficiency_measure_monotonic,
        proficiency_measure_uniform,
    )
    from learner.learners import Learner
    from learner.persistence import (
        persist_abandon_num_submissions,
        persist_abandon_time,
    )

    LEARNER_MODELS_TESTS = {
        "A": (
            Learner(action_space).add_hypothesis(
                proficiency_measure_monotonic(MonotonicUncalibrated, action_space)[0]
            ),
            proficiency_measure_monotonic(MonotonicUncalibrated, action_space)[0],
        ),
        "B": (
            Learner(action_space).add_hypothesis(
                proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0]
            ),
            proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0],
        ),
        "C": (
            Learner(action_space).add_hypothesis(
                persist_abandon_time(MonotonicCalibratedB, action_space)[0]
            ),
            persist_abandon_time(MonotonicCalibratedB, action_space)[0],
        ),
        "C_test_1": (
            Learner(action_space).add_hypothesis(
                persist_abandon_time(MonotonicUncalibrated, action_space)[0]
            ),
            persist_abandon_time(MonotonicUncalibrated, action_space)[0],
        ),
        "C2": (),
        "D": (
            Learner(action_space)
            .add_hypothesis(
                proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0]
            )
            .add_hypothesis(
                proficiency_measure_uniform(
                    UniformDistributionUncalibrated, action_space
                )[0]
            ),
            proficiency_measure_uniform(UniformDistributionUncalibrated, action_space)[
                0
            ],
        ),
        "E": (
            Learner(action_space).add_hypothesis(
                persist_abandon_time(MonotonicCalibratedE, action_space)[0]
            ),
            persist_abandon_time(MonotonicCalibratedE, action_space)[0],
        ),
        "E2": (),
        "F_pre": (
            Learner(action_space)
            .add_hypothesis(
                proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0]
            )
            .add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedF, action_space)[0]
            ),
            proficiency_measure_uniform(UniformCalibratedF, action_space)[0],
        ),
        "F": (
            Learner(action_space)
            .add_hypothesis(
                proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0]
            )
            .add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedF, action_space)[0]
            ),
            proficiency_measure_monotonic(MonotonicCalibratedB, action_space)[0],
        ),
        "G": (
            Learner(action_space).add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedF, action_space)[0]
            ),
            proficiency_measure_uniform(UniformCalibratedF, action_space)[0],
        ),
        "H": (
            Learner(action_space).add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedH, action_space)[0]
            ),
            None,
        ),
        "I": (
            Learner(action_space).add_hypothesis(
                persist_abandon_num_submissions(MonotonicCalibratedI, action_space)[0]
            ),
            None,
        ),
        "J1": (
            Learner(action_space)
            .add_hypothesis(
                persist_abandon_num_submissions(MonotonicCalibratedI, action_space)[0]
            )
            .add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedH, action_space)[0]
            ),
            persist_abandon_num_submissions(MonotonicCalibratedI, action_space)[0],
        ),
        "J2": (
            Learner(action_space)
            .add_hypothesis(
                persist_abandon_num_submissions(MonotonicCalibratedI, action_space)[0]
            )
            .add_hypothesis(
                proficiency_measure_uniform(UniformCalibratedH, action_space)[0]
            ),
            proficiency_measure_uniform(UniformCalibratedH, action_space)[0],
        ),
    }

    result = LEARNER_MODELS_TESTS[tgt_hyp][0].test_hypothesis(
        tgt_hyp_stack=LEARNER_MODELS_TESTS[tgt_hyp][1], dataset_name=dataset_name
    )
    return result


if __name__ == "__main__":
    from environment.action_spaces import HOActionSpaceB, HOActionSpaceC, HOActionSpaceD

    ACTION_SPACES = {
        "B": HOActionSpaceB(),
        "C": HOActionSpaceC(),
        "D": HOActionSpaceD(),
    }

    SUFFIX = "-v2"

    for action_space_label, action_space in ACTION_SPACES.items():
        TGT_HYP = "E"
        config.ACTION_SPACE = action_space
        config.DATASET_NAME = f"node-{TGT_HYP}-actionspace-{action_space_label}{SUFFIX}"
        result = main(
            dataset_name=config.DATASET_NAME, action_space=action_space, tgt_hyp=TGT_HYP
        )
        print(f"Result for Hyp {TGT_HYP} HOActionSpace{action_space_label}: {result}")
        breakpoint()
