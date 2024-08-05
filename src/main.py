import random
from typing import *

import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


from experiments.mdhyp import (
    MonotonicUncalibrated,
    UniformDistributionUncalibrated,
    MonotonicCalibratedB,
    UniformCalibratedF,
    UniformCalibratedH,
    MonotonicCalibratedI,
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
        Learner().add_hypothesis(proficiency_measure_monotonic(MonotonicUncalibrated)),
        proficiency_measure_monotonic(MonotonicUncalibrated),
    ),
    "B": (
        Learner().add_hypothesis(proficiency_measure_monotonic(MonotonicCalibratedB)),
        None,
    ),
    "C": (
        Learner().add_hypothesis(persist_abandon_time(MonotonicCalibratedB)),
        None,
    ),
    "D": (
        Learner()
        .add_hypothesis(proficiency_measure_monotonic(MonotonicCalibratedB))
        .add_hypothesis(proficiency_measure_uniform(UniformDistributionUncalibrated)),
        None,
    ),
    "E": (
        Learner().add_hypothesis(persist_abandon_num_submissions(MonotonicCalibratedB)),
        None,
    ),
    "F": (
        Learner()
        .add_hypothesis(proficiency_measure_monotonic(MonotonicCalibratedB))
        .add_hypothesis(proficiency_measure_uniform(UniformCalibratedF)),
        None,
    ),
    "G": (
        Learner().add_hypothesis(proficiency_measure_uniform(UniformCalibratedF)),
        None,
    ),
    "H": (
        Learner().add_hypothesis(proficiency_measure_uniform(UniformCalibratedH)),
        None,
    ),
    "I": (
        Learner().add_hypothesis(persist_abandon_num_submissions(MonotonicCalibratedI)),
        None,
    ),
    "J": (
        Learner()
        .add_hypothesis(persist_abandon_num_submissions(MonotonicCalibratedI))
        .add_hypothesis(proficiency_measure_uniform(UniformCalibratedH)),
        None,
    ),
}

if __name__ == "__main__":
    TGT_HYP = "A"
    result = LEARNER_MODELS_TESTS[TGT_HYP][0].test_hypothesis(
        LEARNER_MODELS_TESTS[TGT_HYP][1]
    )
    print(result)
