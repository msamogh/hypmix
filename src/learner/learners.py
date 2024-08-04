import copy
import random

random.seed(42)
from dataclasses import dataclass
from enum import Enum
from typing import *

import randomname

from environment.action_spaces import ActionSpace
from environment.state_spaces import StateSweep
from experiments.mdhyp import Hypothesis, MonotonicUncalibrated


class ModelType(Enum):
    THEOR = "theor"
    THEOR_COMP = "theor+comp"
    THEOR_COMP_BEHAV = "theor+comp+behav"
    BEHAV = "behav"


@dataclass
class TheoreticalModel:
    construct_name: str
    definition: str

    def __str__(self):
        return f"{self.construct_name} is defined as '{self.definition}'"


@dataclass
class ComputationalModel:
    construct_name: str
    mappings: dict

    def __str__(self):
        return "\n".join(
            f"{idx + 1}. '{key}' -> {value}"
            for idx, (key, value) in enumerate(self.mappings.items())
        )


@dataclass
class BehavioralModel:
    construct_name: str
    hypotheses: list[Hypothesis]

    def __str__(self):
        return "\n".join([str(h) for h in self.hypotheses])


@dataclass
class SingleHypothesisStack:
    """Encapsulates a single hypothesis with its corresponding theoretical and corresponding models."""

    theoretical_model: TheoreticalModel
    computational_model: ComputationalModel
    hypothesis: Hypothesis


@dataclass
class LearnerCharacteristicModel:
    """Data class for modeling learner characteristics."""

    model_type: ModelType
    theoretical_model: TheoreticalModel
    computational_model: ComputationalModel
    behavioral_model: BehavioralModel

    def __post_init__(self):
        assert (
            self.theoretical_model.construct_name
            == self.computational_model.construct_name
            == self.behavioral_model.construct_name
        ), "Construct names must match in all models."

    def describe(self) -> str:
        """Formats the model descriptions based on its type."""
        if self.model_type == ModelType.BEHAV:
            return str(self.behavioral_model)
        else:
            descriptions = [str(self.theoretical_model)]
            if self.model_type in [ModelType.THEOR_COMP, ModelType.THEOR_COMP_BEHAV]:
                descriptions.append(str(self.computational_model))
            if self.model_type == ModelType.THEOR_COMP_BEHAV:
                descriptions.append(str(self.behavioral_model))
            return "\n\n".join(descriptions)

    def __str__(self):
        return f"{self.theoretical_model.construct_name}.{self.model_type.name}"


@dataclass
class Learner:
    """Data class for a learner."""

    persistence_model: Optional[LearnerCharacteristicModel] = None
    geometry_proficiency_model: Optional[LearnerCharacteristicModel] = None
    persistence_level: Optional[int] = None
    geometry_proficiency_level: Optional[int] = None

    def _check_for_hypothesis_conflicts(self, new_hyp_stack: SingleHypothesisStack):
        """Check if new_hyp contains theoretical or computational models that conflict with the existing learner model.

        Raises an AssertionError if a conflict is found."""
        from learner.geometry_proficiency import (
            THEORETICAL_MODEL_DEFAULT as PROFICIENCY_THEORY,
        )
        from learner.persistence import THEORETICAL_MODEL_DEFAULT as PERSISTENCE_THEORY

        if (
            new_hyp_stack.hypothesis.learner_characteristic
            == PROFICIENCY_THEORY.construct_name
            and self.geometry_proficiency_model is not None
        ):
            existing_model = self.geometry_proficiency_model
        elif (
            new_hyp_stack.hypothesis.learner_characteristic
            == PERSISTENCE_THEORY.construct_name
            and self.persistence_model is not None
        ):
            existing_model = self.persistence_model
        else:
            return
        assert (
            existing_model.theoretical_model == new_hyp_stack.theoretical_model
            and existing_model.computational_model == new_hyp_stack.computational_model
        )

    def _find_behavioral_model(self, learner_characteristic: str):
        from learner.geometry_proficiency import (
            THEORETICAL_MODEL_DEFAULT as PROFICIENCY_THEORY,
        )
        from learner.persistence import THEORETICAL_MODEL_DEFAULT as PERSISTENCE_THEORY

        if learner_characteristic == PROFICIENCY_THEORY.construct_name:
            return self.geometry_proficiency_model.behavioral_model
        elif learner_characteristic == PERSISTENCE_THEORY.construct_name:
            return self.persistence_model.behavioral_model
        raise ValueError(f"Unknown learner characteristic: {learner_characteristic}")

    def add_hypothesis(self, new_hyp: SingleHypothesisStack):
        from learner.geometry_proficiency import (
            THEORETICAL_MODEL_DEFAULT as PROFICIENCY_THEORY,
        )
        from learner.persistence import THEORETICAL_MODEL_DEFAULT as PERSISTENCE_THEORY

        self._check_for_hypothesis_conflicts(new_hyp)

        """Add a new hypothesis to the learner model."""
        if (
            new_hyp.hypothesis.learner_characteristic
            == PROFICIENCY_THEORY.construct_name
        ):
            if self.geometry_proficiency_model is not None:
                # If there's already an existing model, just append the hypothesis to it.
                self.geometry_proficiency_model.behavioral_model.hypotheses.append(
                    new_hyp.hypothesis
                )
            else:
                # Otherwise, create a new model out of the SingleHypothesisStack.
                self.geometry_proficiency_model = LearnerCharacteristicModel(
                    ModelType.THEOR_COMP_BEHAV,
                    new_hyp.theoretical_model,
                    new_hyp.computational_model,
                    BehavioralModel(
                        new_hyp.hypothesis.learner_characteristic, [new_hyp.hypothesis]
                    ),
                )
        elif (
            new_hyp.hypothesis.learner_characteristic
            == PERSISTENCE_THEORY.construct_name
        ):
            if self.persistence_model is not None:
                # If there's already an existing model, just append the hypothesis to it.
                self.persistence_model.behavioral_model.hypotheses.append(
                    new_hyp.hypothesis
                )
            else:
                # Otherwise, create a new model out of the SingleHypothesisStack.
                self.persistence_model = LearnerCharacteristicModel(
                    ModelType.THEOR_COMP_BEHAV,
                    new_hyp.theoretical_model,
                    new_hyp.computational_model,
                    BehavioralModel(
                        new_hyp.hypothesis.learner_characteristic, [new_hyp.hypothesis]
                    ),
                )

        return copy.deepcopy(self)

    def calibrate_hypothesis(
        self,
        hyp_stack: SingleHypothesisStack,
        calibrated_hyp_class: Type[Hypothesis],
    ):
        """Replace an uncalibrated hypothesis in the learner model with its calibrated equivalent.

        Note that this model assumes that the developer is passing in a calibrated calibrated_hyp_class. So it does not do the job of verifying whether the calibrated_hyp_class is calibrated.
        """
        hypothesis = hyp_stack.hypothesis
        # Locate the target hypothesis in the learner model.
        tgt_behavioral_model = self._find_behavioral_model(
            hypothesis.learner_characteristic
        )
        tgt_model_hypotheses = tgt_behavioral_model.hypotheses

        # Replace existing hypothesis from the learner characteristic's BehaviorModel with  the calibrated hypothesis.
        for hyp_idx in range(len(tgt_model_hypotheses)):
            if tgt_model_hypotheses[hyp_idx].behavior_name == hypothesis.behavior_name:
                tgt_behavioral_model.hypotheses = (
                    tgt_model_hypotheses[:hyp_idx]
                    + [
                        calibrated_hyp_class(
                            hypothesis.behavior_name,
                            hypothesis.learner_characteristic,
                        )
                    ]
                    + tgt_model_hypotheses[hyp_idx + 1 :]
                )
                return copy.deepcopy(self)

        # If target_hypothesis is not found in the learner model, throw an error.
        raise ValueError(
            f"No existing hypothesis found for {hypothesis.behavior_name}."
        )

    def remove_hypothesis(self, hyp_stack: SingleHypothesisStack):
        """Remove an existing hypothesis from the learner model."""
        hypothesis = hyp_stack.hypothesis
        tgt_behavioral_model = self._find_behavioral_model(
            hypothesis.learner_characteristic
        )
        tgt_model_hypotheses = tgt_behavioral_model.hypotheses

        # Remove existing hypothesis from the learner characteristic's BehaviorModel
        for hyp_idx in range(len(tgt_model_hypotheses)):
            if tgt_model_hypotheses[hyp_idx].behavior_name == hypothesis.behavior_name:
                tgt_behavioral_model.hypotheses = (
                    tgt_model_hypotheses[:hyp_idx] + tgt_model_hypotheses[hyp_idx + 1 :]
                )
                return copy.deepcopy(self)

        # If target_hypothesis is not found in the learner model, throw an error.
        raise ValueError(
            f"No existing hypothesis found for {hypothesis.behavior_name}."
        )

    def test_hypothesis(
        self,
        tgt_hyp_stack: SingleHypothesisStack,
        dataset_name: Text,
        action_space: ActionSpace,
        state_sweep: StateSweep = None,
        tgt_lc_value_range: Tuple[int, int] = (1, 11),
        fake_llm: bool = False,
        prompt_name: str = "amogh-ld/sl-calibration-1",
        llm_name: Text = "gpt-4-turbo",
        llm_temperature: float = 0.9,
    ):
        """Test whether the target hypothesis is satisfied."""
        from experiments.experiment import Experiment

        from .geometry_proficiency import THEORETICAL_MODEL_DEFAULT as GP_THEORY
        from .persistence import THEORETICAL_MODEL_DEFAULT as P_THEORY

        experiment_outputs = dict()
        # Sweep over different values of the learner characteristic of the target hypothesis.
        for lc_level in range(*tgt_lc_value_range):
            # For the LCs that are not currently being tested, just pick a random value between 1 and 10 (since non-target LCs shouldn't make a difference to the marginal distributional hypothesis).
            if tgt_hypothesis.learner_characteristic == GP_THEORY.construct_name:
                gp_level, persistence_level = lc_level, random.randint(1, 10)
            elif tgt_hypothesis.learner_characteristic == P_THEORY.construct_name:
                persistence_level, gp_level = lc_level, random.randint(1, 10)
            experiment = Experiment(
                experiment_id=randomname.get_name(),
                dataset_name=dataset_name,
                prompt_name=prompt_name,
                geometry_proficiency_model=self.geometry_proficiency_model,
                persistence_model=self.persistence_model,
                geometry_proficiency_levels=[gp_level],
                persistence_levels=[persistence_level],
                state_sweep=state_sweep,
                action_space=action_space,
                model_name=llm_name,
                temperature=llm_temperature,
            )
            experiment_outputs[experiment.experiment_id] = experiment.run(
                fake_llm=fake_llm
            )
        stat, p_value = tgt_hypothesis.statistical_test(experiment_outputs)

        tgt_hypothesis = tgt_hyp_stack.hypothesis
        tgt_behavioral_model = self._find_behavioral_model(
            tgt_hypothesis.learner_characteristic
        )
        tgt_model_hypotheses = tgt_behavioral_model.hypotheses
        for hyp in tgt_model_hypotheses:
            if hyp.behavior_name == tgt_hypothesis.behavior_name:
                return hyp.statistical_test()
