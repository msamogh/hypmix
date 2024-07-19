from typing import *
from enum import Enum
from dataclasses import dataclass

from mdhyp import MDHyp, MonotonicMDHyp1, SlopeMDHyp1


class ModelType(Enum):
    THEOR = "theor"
    THEOR_COMP = "theor+comp"
    THEOR_COMP_BEHAV = "theor+comp+behav"


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
    hypotheses: list[MDHyp]

    def __str__(self):
        hypotheses_str = "\n".join([str(h) for h in self.hypotheses])
        return f"Behavioral Hypotheses: {hypotheses_str}"


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

    persistence_level: int
    geometry_proficiency_level: int
    persistence_model: LearnerCharacteristicModel
    geometry_proficiency_model: LearnerCharacteristicModel


# if __name__ == "__main__":
#     geom_model = create_geometry_proficiency_model(ModelType.BEHAVIORAL)
#     persistence_model = create_persistence_model(ModelType.BEHAVIORAL)
#     print(geom_model.describe())
#     print(persistence_model.describe())
