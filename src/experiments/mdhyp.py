from dataclasses import dataclass, field
from typing import *


@dataclass
class Hypothesis:

    behavior_name: str
    learner_characteristic: str

    def __str__(self):
        raise NotImplementedError

    def test_fn(self, experiment_results):
        raise NotImplementedError


@dataclass
class MonotonicUncalibrated(Hypothesis):
    """A hypothesis class representing all hypotheses about a monotonically-increasing relationship between a learner characteristic and the probability of some desired behavior."""

    behavior_long_description: str
    behavior_actions: List[str]
    positive_relationship: bool

    def __str__(self):

        def actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        if self.positive_relationship:
            return f"A learner with a higher {self.lc_construct.lower()} level is more likely to {self.behavior_name} (i.e., {self.behavior_long_description}). To '{self.behavior_name}' is to make one of the following actions: {actions_list_str()}."
        else:
            return f"A learner with a higher {self.lc_construct.lower()} level is less likely to {self.behavior_name} (i.e., {self.behavior_long_description}). To '{self.behavior_name}' is to make one of the following actions: {actions_list_str()}."

    def test_fn(self, experiment_results):
        from scipy.stats import spearmanr

        x = []
        y = []
        for experiment_id, results in experiment_results.values():
            assert (
                len(results["geometry_proficiency_levels"]) == 1
            ), "Only one geometry proficiency level per experiment is supported for MDHyp1."
            x.append(results["geometry_proficiency_levels"][0])
            y.append(results["productive_actions_ratio"])
        print("Geometry Proficiency Levels:", x)
        print("Action Productive Actions Ratio:", y)
        print("Spearman Correlation Test:")
        correlation, p_value = spearmanr(x, y)
        print(f"Correlation: {correlation}")
        print(f"P-value: {p_value}")
        return correlation, p_value


@dataclass
class UniformDistribution(Hypothesis):
    """A hypothesis about a slope relationship between a learner characteristic and the probability of some desired behavior."""

    def __str__(self):
        raise NotImplementedError

    def test_fn(self, experiment_results):
        raise NotImplementedError
