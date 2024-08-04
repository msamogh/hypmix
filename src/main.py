from typing import *

import random

random.seed(42)

import randomname

from environment.action_spaces import HOActionSpace, HOActionSpaceB
from environment.state_spaces import StateSweep
from experiments.hypothesis_tester import MDHypTester
from experiments.experiment import Experiment
from experiments.mdhyp import Hypothesis
from learner.geometry_proficiency import (
    THEORETICAL_MODEL_DEFAULT as GP_THEORY,
    productive_measurement_monotonic_mdhyp_factory,
    productive_measurement_uniform_mdhyp_factory,
)
from learner.learners import Learner
from learner.persistence import (
    THEORETICAL_MODEL_DEFAULT as P_THEORY,
    abandoning_behavior_num_submissions_mdhyp_factory,
    abandoning_behavior_time_elapsed_mdhyp_factory,
)


def test_hypothesis(
    dataset_name: Text,
    learner: Learner,
    hypothesis: Hypothesis,
    action_space: HOActionSpace,
    state_sweep: StateSweep = None,
    learner_characteristic_value_range: Tuple[int, int] = (1, 11),
    fake_llm: bool = False,
    llm_name: Text = "gpt-4-turbo",
    llm_temperature: float = 0.9,
):
    experiment_results = dict()
    for lc_level in range(*learner_characteristic_value_range):
        if hypothesis.learner_characteristic == GP_THEORY.construct_name:
            gp_level, persistence_level = lc_level, random.randint(1, 10)
        elif hypothesis.learner_characteristic == P_THEORY.construct_name:
            persistence_level, gp_level = lc_level, random.randint(1, 10)
        experiment = Experiment(
            experiment_id=randomname.get_name(),
            dataset_name=dataset_name,
            prompt_name="amogh-ld/sl-calibration-1",
            geometry_proficiency_model=learner.geometry_proficiency_model,
            persistence_model=learner.persistence_model,
            geometry_proficiency_levels=[gp_level],
            persistence_levels=[persistence_level],
            state_sweep=state_sweep,
            action_space=action_space,
            model_name=llm_name,
            temperature=llm_temperature,
        )
        experiment_results[experiment.experiment_id] = experiment.run(fake_llm=fake_llm)


def get_calibrated_class_for_AB():
    # TODO
    raise NotImplementedError


def get_calibrated_class_for_DF():
    # TODO
    raise NotImplementedError


def get_calibrated_class_for_GH():
    # TODO
    raise NotImplementedError


def get_calibrated_class_for_EI():
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    HYPOTHESES = {
        # H_A
        "proficiency_vs_good_measurements": productive_measurement_monotonic_mdhyp_factory(
            action_space=HOActionSpaceB()
        ),
        # H_D
        "unproficient_random_measurements": productive_measurement_uniform_mdhyp_factory(
            action_space=HOActionSpaceB()
        ),
        # H_B
        "persistence_time": abandoning_behavior_time_elapsed_mdhyp_factory(
            action_space=HOActionSpaceB()
        ),
        # H_C
        "persistence_num_submissions": abandoning_behavior_num_submissions_mdhyp_factory(
            action_space=HOActionSpaceB()
        ),
    }

    A = Learner()

    A = A.add_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])

    B = A.calibrate_hypothesis(
        HYPOTHESES["proficiency_vs_good_measurements"],
        get_calibrated_class_for_AB(),
    )

    C = A.remove_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])
    C = C.add_hypothesis(HYPOTHESES["persistence_time"])
    C_result = C.test_hypothesis(HYPOTHESES["persistence_time"])

    D = B.add_hypothesis(HYPOTHESES["unproficient_random_measurements"])

    E = C.remove_hypothesis(HYPOTHESES["persistence_time"])
    E = C.add_hypothesis(HYPOTHESES["persistence_num_submissions"])
    E_result = E.test_hypothesis(HYPOTHESES["persistence_num_submissions"])

    F = D.calibrate_hypothesis(
        HYPOTHESES["unproficient_random_measurements"],
        get_calibrated_class_for_DF(),
    )
    F_result = F.test_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])

    G = F.remove_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])
    G_result = G.test_hypothesis(HYPOTHESES["unproficient_random_measurements"])

    if G_result:
        H = G
    else:
        H = G.calibrate_hypothesis(
            HYPOTHESES["unproficient_random_measurements"],
            get_calibrated_class_for_GH(),
        )

    if E_result:
        I = E
    else:
        I = E.calibrate_hypothesis(
            HYPOTHESES["persistence_num_submissions"],
            get_calibrated_class_for_EI(),
        )

    J = Learner(
        geometry_proficiency_model=H.geometry_proficiency_model,
        persistence_model=I.persistence_model,
    )
    J_result_1 = J.test_hypothesis(
        HYPOTHESES["persistence_num_submissions"],
        get_calibrated_class_for_EI(),
    )
    J_result_2 = J.test_hypothesis(
        HYPOTHESES["unproficient_random_measurements"],
        get_calibrated_class_for_GH(),
    )
