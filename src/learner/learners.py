from dataclasses import dataclass
from enum import Enum
from typing import *
import copy

from experiments.mdhyp import Hypothesis, MonotonicUncalibrated
from learner.geometry_proficiency import (
    _THEORETICAL_MODEL as PROFICIENCY_THEORY,
)
from learner.persistence import _THEORETICAL_MODEL as PERSISTENCE_THEORY


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

    persistence_model: LearnerCharacteristicModel
    geometry_proficiency_model: LearnerCharacteristicModel
    persistence_level: Optional[int] = None
    geometry_proficiency_level: Optional[int] = None

    def _find_behavioral_model(self, learner_characteristic: str):
        if learner_characteristic == PROFICIENCY_THEORY.construct_name:
            return self.geometry_proficiency_model.behavioral_model
        elif learner_characteristic == PERSISTENCE_THEORY.construct_name:
            return self.persistence_model.behavioral_model
        raise ValueError(f"Unknown learner characteristic: {learner_characteristic}")

    def add_hypothesis(self, new_hyp: Hypothesis):
        """Add a new hypothesis to the learner model."""
        if new_hyp.learner_characteristic == PROFICIENCY_THEORY.construct_name:
            self.geometry_proficiency_model.behavioral_model.hypotheses.append(new_hyp)
        elif new_hyp.learner_characteristic == PERSISTENCE_THEORY.construct_name:
            self.persistence_model.behavioral_model.hypotheses.append(new_hyp)
        return copy.deepcopy(self)

    def calibrate_hypothesis(
        self,
        learner_characteristic: str,
        behavior_name: str,
        calibrated_hyp_class: Type[Hypothesis],
    ):
        """Replace an uncalibrated hypothesis in the learner model with its calibrated equivalent.

        Note that this model assumes that the developer is passing in a calibrated calibrated_hyp_class. So it does not do the job of verifying whether the calibrated_hyp_class is calibrated.
        """
        # Locate the target hypothesis in the learner model.
        if learner_characteristic == PROFICIENCY_THEORY.construct_name:
            tgt_model = self.geometry_proficiency_model
        elif learner_characteristic == PERSISTENCE_THEORY.construct_name:
            tgt_model = self.persistence_model
        tgt_model_hypotheses = tgt_model.behavioral_model.hypotheses

        # Replace existing hypothesis from the learner characteristic's BehaviorModel with  the calibrated hypothesis.
        for hyp_idx in range(len(tgt_model_hypotheses)):
            if tgt_model_hypotheses[hyp_idx].behavior_name == behavior_name:
                tgt_model.behavioral_model.hypotheses = (
                    tgt_model_hypotheses[:hyp_idx]
                    + [
                        calibrated_hyp_class(
                            behavior_name,
                            learner_characteristic,
                        )
                    ]
                    + tgt_model_hypotheses[hyp_idx + 1 :]
                )
                return copy.deepcopy(self)

        # If target_hypothesis is not found in the learner model, throw an error.
        raise ValueError(f"No existing hypothesis found for {behavior_name}.")

    def remove_hypothesis(
        self,
        learner_characteristic: str,
        behavior_name: str,
    ):
        """Remove an existing hypothesis from the learner model."""
        if tgt_hyp.learner_characteristic == PROFICIENCY_THEORY.construct_name:
            tgt_model = self.geometry_proficiency_model
        elif tgt_hyp.learner_characteristic == PERSISTENCE_THEORY.construct_name:
            tgt_model = self.persistence_model
        tgt_model_hypotheses = tgt_model.behavioral_model.hypotheses

        return copy.deepcopy(self)

    def test_hypothesis(self, tgt_hyp: Hypothesis):
        """Test whether the target hypothesis is satisfied."""
        pass
