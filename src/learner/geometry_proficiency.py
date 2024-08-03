from typing import *

from .learners import *

from environment.action_spaces import HOActionSpace

_THEORETICAL_MODEL = TheoreticalModel(
    construct_name="Geometry Proficiency",
    definition="The ability to apply the knowledge of the properties of common shapes to solve problems.",
)


def GEOMETRY_PROFICIENCY(
    model_type: ModelType,
    action_space: HOActionSpace,
    monotonic_hyp_class: Type[Hypothesis] = MonotonicUncalibrated,
) -> LearnerCharacteristicModel:
    theoretical_model = (_THEORETICAL_MODEL,)
    computational_model = ComputationalModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        mappings={
            "apply the knowledge of properties of common shapes": "The knowledge that the sum of the distances between any point on an ellipse and its two foci is constant.",
            "to solve problems": "to verify Kepler's First Law",
        },
    )
    productive_measurement_mdhyp = monotonic_hyp_class(
        behavior_name="make productive measurements",
        learner_characteristic=_THEORETICAL_MODEL.construct_name,
        behavior_long_description="those that measure distances between pairs of points in the planetary system that are potentially useful to verify if the orbit is elliptical",
        behavior_actions=action_space.productive_action_labels,
        positive_relationship=True,
    )
    behavioral_model = BehavioralModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        hypotheses=[productive_measurement_mdhyp],
    )
    return LearnerCharacteristicModel(
        model_type,
        theoretical_model,
        computational_model,
        behavioral_model,
    )
