from typing import *
from dataclasses import dataclass, field

import randomname
from scipy.stats import spearmanr

from plot import replot_figures
from experiment import Experiment


@dataclass
class MDHyp:
    experiment_configs: List[Dict[Text, Any]]
    default_values: Dict[Text, Any] = field(
        default_factory=lambda: {
            "model_name": "gpt-4",
            "temperature": 0.9,
            "persistence_levels": None,
        }
    )

    def __post_init__(self):
        self.experiments = []
        for config in self.experiment_configs:
            # Update config with default values
            updated_config = {**self.default_values, **config}
            experiment = Experiment(
                experiment_id=randomname.get_name(), **updated_config
            )
            self.experiments.append(experiment)

    def test(self, fake_llm: bool = False):
        results = {}
        for experiment in self.experiments:
            results[experiment.experiment_id] = experiment.run(fake_llm=fake_llm)
        replot_figures()
        self._test(results)

    def _test(self, results: Dict[Text, Any]):
        raise NotImplementedError


@dataclass
class GeomProductiveMeasureHyp(MDHyp):

    def _test(self, results):
        x = []
        y = []
        for experiment_id, experiment_results in results.items():
            assert (
                len(experiment_results["geometry_proficiency_levels"]) == 1
            ), "Only one geometry proficiency level per experiment is supported for MDHyp1."
            x.append(experiment_results["geometry_proficiency_levels"][0])
            y.append(experiment_results["productive_actions_ratio"])
        print("MDHyp1 Test Results:")
        print("Geometry Proficiency Levels:", x)
        print("Action Productive Actions Ratio:", y)
        print("Spearman Correlation Test:")
        correlation, p_value = spearmanr(x, y)
        print(f"Correlation: {correlation}")
        print(f"P-value: {p_value}")
        return correlation, p_value


@dataclass
class MDHyp2(MDHyp):

    def _test(self, results):
        x = []
        y = []
        for experiment_id, experiment_results in results.items():
            assert (
                len(experiment_results["persistence_levels"]) == 1
            ), "Only one persistence level per experiment is supported for MDHyp2."
            x.append(experiment_results["persistence_levels"][0])
            y.append(experiment_results["exit_percentage"])
        print("MDHyp2 Test Results:")
        print("Persistence Levels:", x)
        print("Exit Percentage:", y)
        print("Spearman Correlation Test:")
        correlation, p_value = spearmanr(x, y)
        print(f"Correlation: {correlation}")
        print(f"P-value: {p_value}")
        return correlation, p_value


@dataclass
class MDHyp3(MDHyp):

    def _test(self, results):
        x = []
        y = []
        for experiment_id, experiment_results in results.items():
            assert (
                len(experiment_results["geometry_proficiency_levels"]) == 1
            ), "Only one geometry proficiency level per experiment is supported for MDHyp3."
            x.append(experiment_results["geometry_proficiency_levels"][0])
            y.append(experiment_results["action_productive_actions_ratio"])
        print("MDHyp3 Test Results:")
        print("Geometry Proficiency Levels:", x)
        print("Action Productive Actions Ratio:", y)
        print("Spearman Correlation")
        correlation, p_value = spearmanr(x, y)
        print(f"Correlation: {correlation}")
        print(f"P-value: {p_value}")
        return correlation, p_value
