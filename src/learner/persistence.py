from typing import *

from .learners import *

from environment.action_spaces import HOActionSpace

_THEORETICAL_MODEL = TheoreticalModel(
    construct_name="Persistence",
    definition="Keeping at a task and finishing it despite the obstacles or the effort involved.",
)


def PERSISTENCE_NUM_SUBMISSIONS(
    model_type: ModelType,
    action_space: HOActionSpace,
    monotonic_hyp_class: Type[Hypothesis] = MonotonicUncalibrated,
) -> LearnerCharacteristicModel:
    computational_model = ComputationalModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        mappings={
            "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
            "finishing it": "Submitting solutions until an acceptable solution is found.",
            "despite the obstacles": "Failed submissions",
            "effort involved": "Number of measurements made or submissions attempted",
        },
    )
    task_abandonment_mdhyp = monotonic_hyp_class(
        learner_characteristic=_THEORETICAL_MODEL.construct_name,
        behavior_name="abandon the task as the number of submissions increases",
        behavior_long_description="to prematurely exit the session before submitting the right solution",
        behavior_actions=[action_space.exit_action_label],
        positive_relationship=False,
    )
    behavioral_model = BehavioralModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        hypotheses=[task_abandonment_mdhyp],
    )
    return LearnerCharacteristicModel(
        model_type,
        theoretical_model=_THEORETICAL_MODEL,
        computational_model=computational_model,
        behavioral_model=behavioral_model,
    )


def PERSISTENCE_TIME_ELAPSED(
    model_type: ModelType,
    action_space: HOActionSpace,
    monotonic_hyp_class: Type[Hypothesis] = MonotonicUncalibrated,
) -> LearnerCharacteristicModel:
    computational_model = ComputationalModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        mappings={
            "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
            "finishing it": "Submitting solutions until an acceptable solution is found.",
            "despite the obstacles": "Failed submissions",
            "effort involved": "Number of measurements made or submissions attempted",
        },
    )
    task_abandonment_mdhyp = monotonic_hyp_class(
        learner_characteristic=_THEORETICAL_MODEL.construct_name,
        behavior_name="abandon the task as the time elapsed increases",
        behavior_long_description="to prematurely exit the session before submitting the right solution",
        behavior_actions=[action_space.exit_action_label],
        positive_relationship=False,
    )
    behavioral_model = BehavioralModel(
        construct_name=_THEORETICAL_MODEL.construct_name,
        hypotheses=[task_abandonment_mdhyp],
    )
    return LearnerCharacteristicModel(
        model_type,
        theoretical_model=_THEORETICAL_MODEL,
        computational_model=computational_model,
        behavioral_model=behavioral_model,
    )
