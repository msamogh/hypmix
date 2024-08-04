import random
from typing import *

random.seed(42)


from environment.action_spaces import HOActionSpaceB
from experiments.mdhyp import (
    MonotonicCalibratedAB,
    UniformCalibratedDF,
    UniformCalibratedGH,
    MonotonicCalibratedEI
)
from learner.geometry_proficiency import (
    productive_measurement_monotonic_mdhyp_factory,
    productive_measurement_uniform_mdhyp_factory,
)
from learner.learners import Learner
from learner.persistence import (
    abandoning_behavior_num_submissions_mdhyp_factory,
    abandoning_behavior_time_elapsed_mdhyp_factory,
)

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
        MonotonicCalibratedAB,
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
        UniformCalibratedDF,
    )
    F_result = F.test_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])

    G = F.remove_hypothesis(HYPOTHESES["proficiency_vs_good_measurements"])
    G_result = G.test_hypothesis(HYPOTHESES["unproficient_random_measurements"])

    if G_result:
        H = G
    else:
        H = G.calibrate_hypothesis(
            HYPOTHESES["unproficient_random_measurements"],
            UniformCalibratedGH,
        )

    if E_result:
        I = E
    else:
        I = E.calibrate_hypothesis(
            HYPOTHESES["persistence_num_submissions"],
            MonotonicCalibratedEI,
        )

    J = Learner(
        geometry_proficiency_model=H.geometry_proficiency_model,
        persistence_model=I.persistence_model,
    )
    J_result_1 = J.test_hypothesis(HYPOTHESES["persistence_num_submissions"])
    J_result_2 = J.test_hypothesis(HYPOTHESES["unproficient_random_measurements"])
