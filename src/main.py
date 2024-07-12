from typing import *

from action_spaces import HOActionSpaceB
from state_spaces import HOStateB, StateSweep
from learners import ModelType
from hypothesis import GeomProductiveMeasureHyp, MDHyp2, MDHyp3


def test_geom_productive_hyp(
    dataset_name: Text,
    state_sweep: StateSweep = None,
    geom_proficiency_range: Tuple[int, int] = (1, 11),
    model_type: ModelType = ModelType.THEOR_COMP_BEHAV,
    fake_llm: bool = False,
    model_name: Text = "gpt-4",
):
    experiment_configs = [
        {
            "dataset_name": dataset_name,
            "prompt_name": "amogh-ld/sl-calibration-1",
            "geometry_proficiency_levels": [geom_proficiency_level],
            "model_types": [model_type],
            "state_sweep": state_sweep,
            "action_space": HOActionSpaceB(),
            "model_name": model_name,
        }
        for geom_proficiency_level in range(*geom_proficiency_range)
    ]
    GeomProductiveMeasureHyp(experiment_configs).test(fake_llm=fake_llm)


def test_mdhyp_2(
    dataset_name: Text,
    state_sweep: StateSweep = None,
    persistence_level_range: Tuple[int, int] = (1, 11),
    model_type: ModelType = ModelType.THEOR_COMP_BEHAV,
):
    experiment_configs = [
        {
            "dataset_name": dataset_name,
            "prompt_name": "amogh-ld/sl-calibration-1",
            "geometry_proficiency_levels": [1],
            "persistence_levels": [persistence_level],
            "model_types": [model_type],
            "state_sweep": state_sweep,
            "action_space": HOActionSpaceB(),
        }
        for persistence_level in range(*persistence_level_range)
    ]
    MDHyp2(experiment_configs).test()


def test_mdhyp_3(
    dataset_name: Text,
    state_sweep: StateSweep = None,
    model_type: ModelType = ModelType.THEOR_COMP_BEHAV,
):
    experiment_configs = [
        {
            "dataset_name": dataset_name,
            "prompt_name": "amogh-ld/sl-calibration-1",
            "geometry_proficiency_levels": [1],
            "persistence_levels": [1],
            "model_types": [model_type],
            "state_sweep": state_sweep,
            "action_space": HOActionSpaceB(),
        }
    ]
    MDHyp3(experiment_configs).test()


if __name__ == "__main__":
    result_1 = test_geom_productive_hyp(
        dataset_name=f"mdhyp1-dataset-108",
        state_sweep=HOStateB.generate_states(),
        geom_proficiency_range=(1, 11),
        model_type=ModelType.THEOR_COMP_BEHAV,
        model_name="gpt-4",
        fake_llm=False,
    )
    # result_2 = test_mdhyp_2(
    #     dataset_name=f"mdhyp2-dataset-1",
    #     state_sweep=HOStateB.generate_toy_state_space(),
    #     persistence_level_range=(2, 4),
    #     model_type=ModelType.THEOR_COMP_BEHAV,
    # )
    # result_3 = test_mdhyp_3(
    #     dataset_name=f"mdhyp3-dataset-1",
    #     state_sweep=HOStateB.generate_toy_state_space(),
    #     model_type=ModelType.THEOR_COMP_BEHAV,
    # )
