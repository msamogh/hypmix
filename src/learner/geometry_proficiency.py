from typing import *

import config
from environment.action_spaces import HOActionSpace
from experiments.mdhyp import MonotonicUncalibrated, UniformDistributionUncalibrated

from .learners import ComputationalModel, SingleHypothesisStack, TheoreticalModel

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


def proficiency_measure_monotonic(
    hyp_class: Type[MonotonicUncalibrated],
    action_space: HOActionSpace = config.ACTION_SPACE,
) -> SingleHypothesisStack:
    return SingleHypothesisStack(
        THEORETICAL_MODEL_DEFAULT,
        COMPUTATIONAL_MODEL_DEFAULT,
        hyp_class(
            behavior_name="productive_measurements_proficiency_increase",
            behavior_description="make productive measurements",
            learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
            behavior_long_description="those that measure distances between pairs of points in the planetary system that are potentially useful to verify if the orbit is elliptical",
            behavior_actions=action_space.productive_action_labels,
            positive_relationship=True,
        ),
    )


def proficiency_measure_uniform(
    hyp_class: Type[UniformDistributionUncalibrated],
    action_space: HOActionSpace = config.ACTION_SPACE,
) -> SingleHypothesisStack:
    return SingleHypothesisStack(
        THEORETICAL_MODEL_DEFAULT,
        COMPUTATIONAL_MODEL_DEFAULT,
        hyp_class(
            behavior_name="all_measurements_equally_likely",
            learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
            behavior_actions=[
                action
                for action in action_space.actions.keys()
                if action_space.is_measure_action(action)
            ],
            low_or_high="low",
        ),
    )
