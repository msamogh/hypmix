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

    def test(self, fake_llm: bool = False):
        experiment_results = {}
        for experiment_config in self.experiment_configs:
            experiment = Experiment(
                experiment_id=randomname.get_name(),
                **experiment_config,
            )
            experiment_results[experiment.experiment_id] = experiment.run(
                fake_llm=fake_llm
            )
        replot_figures()
        self._test(experiment_results)

    def _test(self, results: Dict[Text, Any]):
        pass
