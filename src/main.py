from typing import *

from environment.action_spaces import HOActionSpaceB
from environment.state_spaces import StateSweep
from experiments.hypothesis_tester import MDHypTester
from experiments.mdhyp import Hypothesis
from learner.geometry_proficiency import COMPUTATIONAL_MODEL_DEFAULT as GP_COMP
from learner.geometry_proficiency import THEORETICAL_MODEL_DEFAULT as GP_THEORY
from learner.geometry_proficiency import (
    productive_measurement_monotonic_mdhyp_factory,
    productive_measurement_uniform_mdhyp_factory,
)
from learner.learners import Learner, ModelType
from learner.persistence import COMPUTATIONAL_MODEL_NUM_SUBMISSIONS as P_COMP_N
from learner.persistence import COMPUTATIONAL_MODEL_TIME_ELAPSED as P_COMP_T
from learner.persistence import THEORETICAL_MODEL_DEFAULT as P_THEORY
from learner.persistence import (
    abandoning_behavior_num_submissions_mdhyp_factory,
    abandoning_behavior_time_elapsed_mdhyp_factory,
)


def test_geom_productive_hyp(
    dataset_name: Text,
    state_sweep: StateSweep = None,
    learner_characteristic_value_range: Tuple[int, int] = (1, 11),
    fake_llm: bool = False,
    model_name: Text = "gpt-4-turbo",
):
    experiment_configs = [
        {
            "dataset_name": dataset_name,
            "prompt_name": "amogh-ld/sl-calibration-1",
            "geometry_proficiency_model": GEOMETRY_PROFICIENCY_MONO(
                ModelType.THEOR_COMP_BEHAV
            ),
            "persistence_model": None,
            "geometry_proficiency_levels": [geom_proficiency_level],
            "state_sweep": state_sweep,
            "action_space": HOActionSpaceB(),
            "model_name": model_name,
        }
        for geom_proficiency_level in range(*learner_characteristic_value_range)
    ]
    MDHypTester(experiment_configs).test(fake_llm=fake_llm)


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
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.learner_characteristic,
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.behavior_name,
        get_calibrated_class_for_AB(),
    )

    C = A.remove_hypothesis(
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.learner_characteristic,
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.behavior_name,
    )
    C = C.add_hypothesis(HYPOTHESES["persistence_time"])
    C_result = C.test_hypothesis(
        HYPOTHESES[
            "persistence_time"
        ].hypothesis.learner_characteristic,
        HYPOTHESES["persistence_time"].hypothesis.behavior_name,
    )

    D = B.add_hypothesis(HYPOTHESES["unproficient_random_measurements"])

    E = C.remove_hypothesis(
        HYPOTHESES[
            "persistence_time"
        ].hypothesis.learner_characteristic,
        HYPOTHESES["persistence_time"].hypothesis.behavior_name,
    )
    E = C.add_hypothesis(
        HYPOTHESES["persistence_num_submissions"]
    )
    E_result = E.test_hypothesis(
        HYPOTHESES[
            "persistence_num_submissions"
        ].hypothesis.learner_characteristic,
        HYPOTHESES["persistence_num_submissions"].hypothesis.behavior_name,
    )

    F = D.calibrate_hypothesis(
        HYPOTHESES["unproficient_random_measurements"].hypothesis.learner_characteristic,
        HYPOTHESES["unproficient_random_measurements"].hypothesis.behavior_name,
        get_calibrated_class_for_DF(),
    )
    F_result = F.test_hypothesis(
        HYPOTHESES["proficiency_vs_good_measurements"]
    )

    G = F.remove_hypothesis(
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.learner_characteristic,
        HYPOTHESES["proficiency_vs_good_measurements"].hypothesis.behavior_name,
    )
    G_result = G.test_hypothesis(
        HYPOTHESES[
            "unproficient_random_measurements"
        ].hypothesis.learner_characteristic,
        HYPOTHESES["unproficient_random_measurements"].hypothesis.behavior_name,
    )

    if G_result:
        H = G
    else:
        H = G.calibrate_hypothesis(
            HYPOTHESES["unproficient_random_measurements"].hypothesis.learner_characteristic,
            HYPOTHESES["unproficient_random_measurements"].hypothesis.behavior_name,
            get_calibrated_class_for_GH()
        )

    if E_result:
        I = E
    else:
        I = E.calibrate_hypothesis(
            HYPOTHESES["persistence_num_submissions"].hypothesis.learner_characteristic,
            HYPOTHESES["persistence_num_submissions"].hypothesis.behavior_name,
            get_calibrated_class_for_EI()
        )

    J = Learner(
        geometry_proficiency_model=H.geometry_proficiency_model,
        persistence_model=I.persistence_model
    )
    J_result_1 = J.test_hypothesis(
        HYPOTHESES["persistence_num_submissions"].hypothesis.learner_characteristic,
        HYPOTHESES["persistence_num_submissions"].hypothesis.behavior_name,
        get_calibrated_class_for_EI()
    )
    J_result_2 = J.test_hypothesis(
        HYPOTHESES["unproficient_random_measurements"].hypothesis.learner_characteristic,
        HYPOTHESES["unproficient_random_measurements"].hypothesis.behavior_name,
        get_calibrated_class_for_GH()
    )
