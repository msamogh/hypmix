from dataclasses import dataclass, field
from typing import *


@dataclass(eq=True)
class Hypothesis:

    behavior_name: str
    learner_characteristic: str

    def __str__(self):
        raise NotImplementedError

    def statistical_test(self, experiment_set_results) -> Tuple[float, float]:
        """Returns a statistic and a p-value."""
        raise NotImplementedError
    
    def __eq__(self, other):
        return self.behavior_name == other.behavior_name and self.learner_characteristic and other.learner_characteristic


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

    def test_fn(
        self,
        experiment_set_results,
        lc_key: str = "geometry_proficiency_levels",
        tgt_action_label_key: str = "productive_actions_ratio",
    ):
        from scipy.stats import spearmanr

        x = []
        y = []
        for experiment_id, experiment_results in experiment_set_results.values():
            assert (
                len(experiment_results[lc_key]) == 1
            ), "Only one LC level per experiment is supported for Monotonic hypotheses."
            x.append(experiment_results[lc_key][0])
            y.append(experiment_results[tgt_action_label_key])
        print("LC Levels:", x)
        print("Target Action Ratio:", y)
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

    def test_fn(self, experiment_set_results, tgt_action_labels):
        from scipy.stats import chisquare

        assert (
            len(experiment_set_results) == 1
        ), "Uniform distribution tests can be run only on a single experiment, not an experiment set."

        # Dummy loop (since there is only one iteration anyway)
        for _, experiment_results in experiment_set_results.values():
            return chisquare(
                [experiment_results[action_label] for action_label in tgt_action_labels]
            )
        raise RuntimeError("Uniform distribution could not be tested.")


@dataclass
class MonotonicCalibratedAB(MonotonicUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class MonotonicCalibratedEI(MonotonicUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class UniformCalibratedDF(UniformDistributionUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class UniformCalibratedGH(UniformDistributionUncalibrated):

    def __str__(self):
        raise NotImplementedError
