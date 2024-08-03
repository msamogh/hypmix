from typing import *

from environment.action_spaces import HOActionSpaceB
from environment.state_spaces import HOStateB, StateSweep
from experiments.hypothesis_tester import MDHypTester, test_hypothesis
from experiments.mdhyp import Hypothesis
from learner.geometry_proficiency import GEOMETRY_PROFICIENCY
from learner.learners import Learner, LearnerCharacteristicModel, ModelType
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


learner_models = [(GEOMETRY_PROFICIENCY, None), ()]


if __name__ == "__main__":
    test_hypothesis()
    # H_A_test_result = test_hypothesis(
    #     dataset_name="mdhyp1-dataset-116",
    #     state_sweep=HOStateB.generate_toy_state_space(num_samples=10),
    #     geom_proficiency_range=(1, 11),
    #     fake_llm=False,
    # )
