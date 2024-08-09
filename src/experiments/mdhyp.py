from dataclasses import dataclass, field
from typing import *
from typing import Tuple

import config
from environment.action_spaces import ActionSpace
from environment.state_spaces import StateSweep


@dataclass
class Hypothesis:

    behavior_name: str
    learner_characteristic: str
    action_space: ActionSpace
    stat_test_kwargs: Dict[Text, Any]

    @property
    def state_sweep(self):
        raise NotImplementedError

    @property
    def is_multi_run_hyp(self):
        raise NotImplementedError

    @property
    def tgt_lc_value_range(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def statistical_test(self, experiment_set_results) -> Tuple[float, float]:
        """Returns a statistic and a p-value."""
        raise NotImplementedError


@dataclass
class MonotonicUncalibrated(Hypothesis):
    """Hypothesis class of all monotonically-increasing relationships between a learner characteristic and the probability of a target action."""

    behavior_description: str
    behavior_long_description: str
    behavior_actions: List[str]
    positive_relationship: bool

    @property
    def state_sweep(self):
        return config.STATE_SWEEP_MED

    @property
    def is_multi_run_hyp(self):
        return True

    @property
    def tgt_lc_value_range(self):
        return (1, 11)

    def __str__(self):

        def actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        if self.positive_relationship:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is more likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {actions_list_str()}."
        else:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is less likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {actions_list_str()}."

    def statistical_test(self, experiment_set_results):
        from scipy.stats import spearmanr

        lc_key = self.stat_test_kwargs["lc_key"]
        tgt_action_label_key = self.stat_test_kwargs["tgt_metric_key"]

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

    @property
    def state_sweep(self):
        return config.STATE_SWEEP_UNIFORM_1

    @property
    def is_multi_run_hyp(self):
        return False

    @property
    def tgt_lc_value_range(self):
        return (1, 2)

    def __post_init__(self):
        assert self.low_or_high in ["low", "high"]

    def __str__(self):

        def behavior_actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        return f"As learners get closer and closer to the {self.low_or_high}er end of the {self.learner_characteristic.lower()} spectrum (value of {1 if self.low_or_high == 'low' else 'high'}) are equally likely to perform the following actions. In other words, such a learner exhibits a uniform distribution over these actions: {behavior_actions_list_str()}"

    def statistical_test(self, experiment_set_results):
        import numpy as np
        from scipy.stats import chisquare

        tgt_action_labels = self.behavior_actions

        assert (
            len(experiment_set_results) == 1
        ), "Uniform distribution tests can be run only on a single experiment, not an experiment set."

        values = experiment_set_results["next_actions"]
        unique_values, observed_frequencies = np.unique(values, return_counts=True)
        expected_frequency = len(values) / len(unique_values)
        expected_frequencies = np.full(len(unique_values), expected_frequency)
        return chisquare(observed_frequencies, f_exp=expected_frequencies)


@dataclass
class MonotonicCalibratedB(MonotonicUncalibrated):

    def __str__(self):

        def behavior_actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        if self.positive_relationship:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is more likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {behavior_actions_list_str()}. In the event that your commonsense reasoning DIRECTLY conflicts with this hypothesis, use this hypothesis."
        else:
            return f"A learner with a higher {self.learner_characteristic.lower()} level is less likely to {self.behavior_description} (i.e., {self.behavior_long_description}). To '{self.behavior_description}' is to make one of the following actions: {behavior_actions_list_str()}. In the event that your commonsense reasoning DIRECTLY conflicts with this hypothesis, use this hypothesis."


@dataclass
class MonotonicCalibratedI(MonotonicUncalibrated):

    def __str__(self):
        raise NotImplementedError


@dataclass
class UniformCalibratedF(UniformDistributionUncalibrated):

    def __str__(self):

        def behavior_actions_list_str():
            return ", ".join([f"'{action}'" for action in self.behavior_actions])

        return f"We know from research that learners with {self.learner_characteristic.lower()} of 1 are known to mindlessly pick a random action from the following set of actions: {behavior_actions_list_str()}"


@dataclass
class UniformCalibratedH(UniformDistributionUncalibrated):

    def __str__(self):
        raise NotImplementedError
