import random
from typing import *

import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import config
from environment.action_spaces import HOActionSpaceB, HOActionSpaceC, HOActionSpaceD
from experiments.mdhyp import (
    MonotonicCalibratedB,
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
from learner.persistence import persist_abandon_num_submissions, persist_abandon_time

LEARNER_MODELS_TESTS = {
    "A": (
        Learner().add_hypothesis(
            proficiency_measure_monotonic(MonotonicUncalibrated)[0]
        ),
        proficiency_measure_monotonic(MonotonicUncalibrated)[0],
    ),
    "B": (
        Learner().add_hypothesis(
            proficiency_measure_monotonic(MonotonicCalibratedB)[0]
        ),
        proficiency_measure_monotonic(MonotonicCalibratedB)[0],
    ),
    "C": (
        Learner().add_hypothesis(persist_abandon_time(MonotonicCalibratedB)[0]),
        persist_abandon_time(MonotonicCalibratedB)[0],
    ),
    "C_test_1": (
        Learner().add_hypothesis(persist_abandon_time(MonotonicUncalibrated)[0]),
        persist_abandon_time(MonotonicUncalibrated)[0],
    ),
    "C2": (),
    "D": (
        Learner()
        .add_hypothesis(proficiency_measure_monotonic(MonotonicCalibratedB)[0])
        .add_hypothesis(
            proficiency_measure_uniform(UniformDistributionUncalibrated)[0]
        ),
        None,
    ),
    "E": (
        Learner().add_hypothesis(
            persist_abandon_num_submissions(MonotonicCalibratedB)[0]
        ),
        persist_abandon_num_submissions(MonotonicCalibratedB)[0],
    ),
    "E2": (),
    "F": (
        Learner()
        .add_hypothesis(proficiency_measure_monotonic(MonotonicCalibratedB)[0])
        .add_hypothesis(proficiency_measure_uniform(UniformCalibratedF)[0]),
        proficiency_measure_monotonic(MonotonicCalibratedB)[0],
    ),
    "G": (
        Learner().add_hypothesis(proficiency_measure_uniform(UniformCalibratedF)[0]),
        proficiency_measure_uniform(UniformCalibratedF)[0],
    ),
    "H": (
        Learner().add_hypothesis(proficiency_measure_uniform(UniformCalibratedH)[0]),
        None,
    ),
    "I": (
        Learner().add_hypothesis(
            persist_abandon_num_submissions(MonotonicCalibratedI)[0]
        ),
        None,
    ),
    "J1": (
        Learner()
        .add_hypothesis(persist_abandon_num_submissions(MonotonicCalibratedI)[0])
        .add_hypothesis(proficiency_measure_uniform(UniformCalibratedH)[0]),
        persist_abandon_num_submissions(MonotonicCalibratedI)[0],
    ),
    "J2": (
        Learner()
        .add_hypothesis(persist_abandon_num_submissions(MonotonicCalibratedI)[0])
        .add_hypothesis(proficiency_measure_uniform(UniformCalibratedH)[0]),
        proficiency_measure_uniform(UniformCalibratedH)[0],
    ),
}

if __name__ == "__main__":
    TGT_HYP = "A"
    result = LEARNER_MODELS_TESTS[TGT_HYP][0].test_hypothesis(
        LEARNER_MODELS_TESTS[TGT_HYP][1]
    )
    print(f"Result: {result}")
