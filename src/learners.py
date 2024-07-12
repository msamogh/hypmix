from enum import Enum
from dataclasses import dataclass


class ModelType(Enum):
    THEOR = "theor"
    THEOR_COMP = "theor+comp"
    THEOR_COMP_BEHAV = "theor+comp+behav"


@dataclass
class TheoreticalModel:
    description: str

    def __str__(self):
        return self.description


@dataclass
class ComputationalModel:
    mappings: dict

    def __str__(self):
        return "\n".join(
            f"{idx + 1}. '{key}' -> {value}"
            for idx, (key, value) in enumerate(self.mappings.items())
        )


@dataclass
class BehavioralModel:
    hypothesis: str

    def __str__(self):
        return f"Behavioral Hypothesis: {self.hypothesis}"


@dataclass
class LearnerCharacteristicModel:
    """Data class for modeling learner characteristics."""

    model_type: ModelType
    theoretical_model: TheoreticalModel
    computational_model: ComputationalModel
    behavioral_model: BehavioralModel

    def describe(self) -> str:
        """Formats the model descriptions based on its type."""
        descriptions = [str(self.theoretical_model)]
        if self.model_type in [ModelType.THEOR_COMP, ModelType.THEOR_COMP_BEHAV]:
            descriptions.append(str(self.computational_model))
        if self.model_type == ModelType.THEOR_COMP_BEHAV:
            descriptions.append(str(self.behavioral_model))
        return "\n\n".join(descriptions)


# Define configurations for specific characteristics
def create_geometry_proficiency_model(
    model_type: ModelType,
) -> LearnerCharacteristicModel:
    theoretical_description = "'Geometry Proficiency' is defined as 'The ability to apply the knowledge of the properties of common shapes to solve problems.'"
    computational_mappings = {
        "apply the knowledge of properties of common shapes": "The knowledge that the sum of the distances between any point on an ellipse and its two foci is constant.",
        "to solve problems": "to verify Kepler's First Law",
    }
    behavioral_hypothesis = "Concretely, a learner with a higher geometry proficiency has a lower likelihood of making 'unproductive' measurements. 'Productive' measurements are only those actions that measure distances between pairs of points in the planetary system that are potentially useful to verify if the orbit is elliptical. These include: MEASURE-F1-Oi, MEASURE-F2-Oi, MEASURE-A-F1, MEASURE-A-F2, MEASURE-F1-P, MEASURE-F2-P. The rest of the measurements are considered 'unproductive'."
    return LearnerCharacteristicModel(
        model_type,
        TheoreticalModel(theoretical_description),
        ComputationalModel(computational_mappings),
        BehavioralModel(behavioral_hypothesis),
    )


def create_persistence_model(model_type: ModelType) -> LearnerCharacteristicModel:
    theoretical_description = "'Persistence' is defined as 'Keeping at a task and finishing it despite the obstacles or the effort involved.'"
    computational_mappings = {
        "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
        "finishing it": "Submitting solutions until an acceptable solution is found.",
        "despite the obstacles": "Failed submissions",
        "effort involved": "Number of measurements made or submissions attempted",
    }
    behavioral_hypothesis = "Concretely, a learner with a higher persistence level is less likely to abandon the session by choosing the action EXIT before the right solution is submitted."
    return LearnerCharacteristicModel(
        model_type,
        TheoreticalModel(theoretical_description),
        ComputationalModel(computational_mappings),
        BehavioralModel(behavioral_hypothesis),
    )


def create_persistence_model_2(model_type: ModelType) -> LearnerCharacteristicModel:
    theoretical_description = "'Persistence' is defined as 'Keeping at a task and finishing it despite the obstacles or the effort involved.'"
    computational_mappings = {
        "keeping at a task": "Continuing to work on the task by making measurements and attempting submissions.",
        "finishing it": "Submitting solutions until an acceptable solution is found.",
        "despite the obstacles": "Failed submissions",
        "effort involved": "Number of measurements made or submissions attempted",
    }
    behavioral_hypothesis = "Concretely, a learner with a higher persistence level is less likely to abandon the session by choosing the action EXIT before the right solution is submitted."
    return LearnerCharacteristicModel(
        model_type,
        TheoreticalModel(theoretical_description),
        ComputationalModel(computational_mappings),
        BehavioralModel(behavioral_hypothesis),
    )


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
