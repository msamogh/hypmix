from typing import *

from environment.action_spaces import HOActionSpaceB
from environment.state_spaces import StateSweep
from experiments.hypothesis_tester import MDHypTester
from learner.geometry_proficiency import GEOMETRY_PROFICIENCY
from learner.learners import Learner, ModelType
from learner.persistence import PERSISTENCE_NUM_SUBMISSIONS, PERSISTENCE_TIME_ELAPSED


def test_geom_productive_hyp(
    dataset_name: Text,
    state_sweep: StateSweep = None,
    geom_proficiency_range: Tuple[int, int] = (1, 11),
    model_type: ModelType = ModelType.BEHAV,
    fake_llm: bool = False,
    model_name: Text = "gpt-4-turbo",
):
    experiment_configs = [
        {
            "dataset_name": dataset_name,
            "prompt_name": "amogh-ld/sl-calibration-1",
            "geometry_proficiency_model": GEOMETRY_PROFICIENCY(model_type),
            "persistence_model": None,
            "geometry_proficiency_levels": [geom_proficiency_level],
            "state_sweep": state_sweep,
            "action_space": HOActionSpaceB(),
            "model_name": model_name,
        }
        for geom_proficiency_level in range(*geom_proficiency_range)
    ]
    MDHypTester(experiment_configs).test(fake_llm=fake_llm)


if __name__ == "__main__":
    G = GEOMETRY_PROFICIENCY(
        model_type=ModelType.THEOR_COMP_BEHAV, action_space=HOActionSpaceB()
    )
    P_t = PERSISTENCE_TIME_ELAPSED(
        model_type=ModelType.THEOR_COMP_BEHAV, action_space=HOActionSpaceB()
    )
    P_n = PERSISTENCE_NUM_SUBMISSIONS(
        model_type=ModelType.THEOR_COMP_BEHAV, action_space=HOActionSpaceB()
    )

    A = Learner()
    A = A.add_hypothesis()

    B = A.calibrate_hypothesis()

    C = A.remove_hypothesis()
    C = C.add_hypothesis()
    C_result = C.test_hypothesis()

    D = B.add_hypothesis()

    E = C.remove_hypothesis()
    E = C.add_hypothesis()
    E_result = E.test_hypothesis()

    F = D.calibrate_hypothesis()
    F_result = F.test_hypothesis()

    G = F.remove_hypothesis()
    G_result = G.test_hypothesis()

    if G_result:
        H = G
    else:
        H = G.calibrate_hypothesis()

    if E_result:
        I = E
    else:
        I = E.calibrate_hypothesis()

    J = Learner()
    J.add_hypothesis()
    J.add_hypothesis()
    J_result = J.test_hypothesis()
