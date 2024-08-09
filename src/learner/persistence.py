from typing import *

import config
from environment.action_spaces import HOActionSpace
from experiments.mdhyp import Hypothesis, MonotonicUncalibrated

from .learners import ComputationalModel, SingleHypothesisStack, TheoreticalModel

THEORETICAL_MODEL_DEFAULT = TheoreticalModel(
    construct_name="Persistence",
    definition="Keeping at a task and finishing it despite the obstacles or the effort involved.",
)

THEORETICAL_MODEL_INVITE = TheoreticalModel(
    construct_name="Persistence",
    definition="Maintaining a sustained effort toward completion of a goal-directed task despite challenges or difficulties",
)

COMPUTATIONAL_MODEL_NUM_SUBMISSIONS = ComputationalModel(
    construct_name=THEORETICAL_MODEL_DEFAULT.construct_name,
    mappings={
        "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
        "finishing it": "Submitting solutions until an acceptable solution is found.",
        "despite the obstacles": "Failed submissions",
        "effort involved": "Number of measurements made or submissions attempted",
    },
)

COMPUTATIONAL_MODEL_TIME_ELAPSED = ComputationalModel(
    construct_name=THEORETICAL_MODEL_DEFAULT.construct_name,
    mappings={
        "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
        "finishing it": "Submitting solutions until an acceptable solution is found.",
        "despite the obstacles": "Failed submissions",
        "effort involved": "Number of measurements made or submissions attempted",
    },
)


def persist_abandon_num_submissions(
    hyp_class: Type[MonotonicUncalibrated],
    action_space: HOActionSpace,
) -> list[SingleHypothesisStack]:
    return [
        SingleHypothesisStack(
            THEORETICAL_MODEL_DEFAULT,
            COMPUTATIONAL_MODEL_NUM_SUBMISSIONS,
            hyp_class(
                behavior_name="task_abandon_num_submissions_increase",
                learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
                action_space=action_space,
                stat_test_kwargs={
                    "lc_key": "persistence_levels",
                    "tgt_metric_key": f"{action_space.exit_action_label.lower()}_percentage",
                },
                behavior_description="abandon the task as the number of submissions increases",
                behavior_long_description="to prematurely exit the session before submitting the right solution",
                behavior_actions=[action_space.exit_action_label],
                positive_relationship=False,
            ),
        ),
        SingleHypothesisStack(
            THEORETICAL_MODEL_DEFAULT,
            COMPUTATIONAL_MODEL_NUM_SUBMISSIONS,
            hyp_class(
                behavior_name="task_abandon_num_submissions_increase",
                learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
                action_space=action_space,
                stat_test_kwargs={
                    "lc_key": "num_submissions",
                    "tgt_metric_key": "quit_percentage",
                },
                behavior_description="abandon the task as the number of submissions increases",
                behavior_long_description="to prematurely exit the session before submitting the right solution",
                behavior_actions=[action_space.exit_action_label],
                positive_relationship=False,
            ),
        ),
    ]


def persist_abandon_time(
    hyp_class: Type[MonotonicUncalibrated],
    action_space: HOActionSpace,
) -> SingleHypothesisStack:
    return [
        SingleHypothesisStack(
            THEORETICAL_MODEL_DEFAULT,
            COMPUTATIONAL_MODEL_TIME_ELAPSED,
            hyp_class(
                behavior_name="task_abandon_time_increase",
                learner_characteristic=THEORETICAL_MODEL_DEFAULT.construct_name,
                action_space=action_space,
                stat_test_kwargs={
                    "lc_key": "persistence_levels",
                    "tgt_metric_key": "quit_percentage",
                },
                behavior_description="abandon the task as the time elapsed increases",
                behavior_long_description="to prematurely exit the session before submitting the right solution",
                behavior_actions=[action_space.exit_action_label],
                positive_relationship=False,
            ),
        )
    ]
