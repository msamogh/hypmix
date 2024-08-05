from dataclasses import dataclass, field
from typing import *
from typing import Tuple


@dataclass
class Hypothesis:

    behavior_name: str
    learner_characteristic: str

    def __str__(self):
        raise NotImplementedError

    def statistical_test(
        self, experiment_set_results, **stat_test_kwargs
    ) -> Tuple[float, float]:
        """Returns a statistic and a p-value."""
        raise NotImplementedError


@dataclass
class MonotonicUncalibrated(Hypothesis):
    """Hypothesis class of all monotonically-increasing relationships between a learner characteristic and the probability of a target action."""

    behavior_description: str
    behavior_long_description: str
    behavior_actions: List[str]
    positive_relationship: bool

    def __str__(self):

        def actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        if self.positive_relationship:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is more likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {actions_list_str()}."
        else:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is less likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {actions_list_str()}."

    def statistical_test(self, experiment_set_results, **stat_test_kwargs):
        from scipy.stats import spearmanr

        lc_key = stat_test_kwargs.get("lc_key", "geometry_proficiency_levels")
        tgt_action_label_key = stat_test_kwargs.get(
            "tgt_metric_key", "productive_actions_ratio"
        )

        x = []
        y = []
        for experiment_results in experiment_set_results.values():
            assert (
                len(experiment_results[lc_key]) == 1
            ), "Only one LC level per experiment is supported for Monotonic hypotheses."
            x.append(experiment_results[lc_key][0])
            y.append(experiment_results[tgt_action_label_key])
        print("LC Levels:", x)
        print("Target Metrics:", y)
        print("Spearman Correlation Test:")
        correlation, p_value = spearmanr(x, y)
        print(f"Correlation: {correlation}")
        print(f"P-value: {p_value}")
        return correlation, p_value


@dataclass
class UniformDistributionUncalibrated(Hypothesis):
    """Hypothesis class of uniform probability distributions of a target action with respect to a learner characteristic."""

    behavior_actions: List[str]
    low_or_high: str

    def __post_init__(self):
        assert self.low_or_high in ["low", "high"]

    def __str__(self):

        def actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        return f"As learners get closer and closer to the {self.low_or_high}er end of the {self.learner_characteristic.lower()} spectrum (value of {1 if self.low_or_high == 'low' else 'high'}) are equally likely to perform the following actions. In other words, such a learner exhibits a uniform distribution over these actions: {actions_list_str()}"

    def statistical_test(self, experiment_set_results, **stat_test_kwargs):
        from scipy.stats import chisquare

        tgt_action_labels = stat_test_kwargs.get("tgt_action_labels", None)

        assert (
            len(experiment_set_results) == 1
        ), "Uniform distribution tests can be run only on a single experiment, not an experiment set."

        # Dummy loop (since there is only one iteration anyway)
        for experiment_results in experiment_set_results.values():
            return chisquare(
                [experiment_results[action_label] for action_label in tgt_action_labels]
            )
        raise RuntimeError("Uniform distribution could not be tested.")


@dataclass
class MonotonicCalibratedB(MonotonicUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class MonotonicCalibratedI(MonotonicUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class UniformCalibratedF(UniformDistributionUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class UniformCalibratedH(UniformDistributionUncalibrated):

    def __str__(self):
        raise NotImplementedError
