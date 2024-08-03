from dataclasses import dataclass, field
from typing import *

import randomname
from scipy.stats import spearmanr

from learner.learners import LearnerCharacteristicModel, Learner

from .experiment import Experiment
from .mdhyp import Hypothesis
from .plot import replot_figures


@dataclass
class MDHypTester:
    experiment_configs: List[Dict[Text, Any]]
    default_experiment_config: Dict[Text, Any] = field(
        default_factory=lambda: {
            "model_name": "gpt-4-turbo",
            "temperature": 0.9,
            "persistence_levels": None,
            "prompt_name": "amogh-ld/sl-calibration-1",
        }
    )

    def test(self, fake_llm: bool = False):
        experiment_results = {}
        for experiment_config in self.experiment_configs:
            experiment = Experiment(
                experiment_id=randomname.get_name(),
                **{**self.default_experiment_config, **experiment_config},
            )
            experiment_results[experiment.experiment_id] = experiment.run(
                fake_llm=fake_llm
            )
        replot_figures()
        self._test(experiment_results)

    def _test(self, results: Dict[Text, Any]):
        pass


def test_hypothesis(
    lc_model_geom_proficiency: LearnerCharacteristicModel,
    lc_model_persistence: LearnerCharacteristicModel,
    test_hyp: Hypothesis,
    **config_kwargs,
):
    """WIP function to test a hypothesis."""

    # Populate a "_range" variable that defines  the sweep range of the learner characteristic being tested.
    if (
        test_hyp.learner_characteristic
        == lc_model_geom_proficiency.theoretical_model.construct_name
    ):
        geom_proficiency_range = config_kwargs.pop("geom_proficiency_range")
        experiment_configs = [
            config_kwargs
            | {
                "persistence_model": lc_model_persistence,
                "geometry_proficiency_model": lc_model_geom_proficiency,
                "geometry_proficiency_levels": [geom_proficiency_level],
            }
            for geom_proficiency_level in range(*geom_proficiency_range)
        ]
    elif (
        test_hyp.learner_characteristic
        == lc_model_persistence.theoretical_model.construct_name
    ):
        persistence_range = config_kwargs.pop("persistence_range")
        experiment_configs = [
            config_kwargs
            | {
                "geometry_proficiency_model": lc_model_geom_proficiency,
                "persistence_model": lc_model_persistence,
                "persistence_levels": [persistence_level],
            }
            for persistence_level in range(*persistence_range)
        ]
    else:
        raise RuntimeError(
            f"Learner characteristic {test_hyp.learner_characteristic} not supported"
        )

    MDHypTester(experiment_configs).test(fake_llm=config_kwargs["fake_llm"])
