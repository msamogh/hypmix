from typing import *

from environment.action_spaces import HOActionSpace
from experiments.mdhyp import (
    Hypothesis,
    MonotonicUncalibrated,
    UniformDistributionUncalibrated,
)
from .learners import TheoreticalModel, ComputationalModel, SingleHypothesisStack

THEORETICAL_MODEL_DEFAULT = TheoreticalModel(
    construct_name="Geometry Proficiency",
    definition="The ability to apply the knowledge of the properties of common shapes to solve problems.",
)

COMPUTATIONAL_MODEL_DEFAULT = ComputationalModel(
    construct_name=THEORETICAL_MODEL_DEFAULT.construct_name,
    mappings={
        "apply the knowledge of properties of common shapes": "The knowledge that the sum of the distances between any point on an ellipse and its two foci is constant.",
        "to solve problems": "to verify Kepler's First Law",
    },
)


def productive_measurement_monotonic_mdhyp_factory(
    action_space: HOActionSpace,
) -> SingleHypothesisStack:
    return SingleHypothesisStack(
        THEORETICAL_MODEL_DEFAULT,
        COMPUTATIONAL_MODEL_DEFAULT,
        MonotonicUncalibrated(
            behavior_name="productive_measurements_proficiency_increase",
            behavior_description="make productive measurements",
            learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
            behavior_long_description="those that measure distances between pairs of points in the planetary system that are potentially useful to verify if the orbit is elliptical",
            behavior_actions=action_space.productive_action_labels,
            positive_relationship=True,
        ),
    )


def productive_measurement_uniform_mdhyp_factory(
    action_space: HOActionSpace,
) -> SingleHypothesisStack:
    return SingleHypothesisStack(
        THEORETICAL_MODEL_DEFAULT,
        COMPUTATIONAL_MODEL_DEFAULT,
        UniformDistributionUncalibrated(
            behavior_name="all_measurements_equally_likely",
            learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
        ),
    )
