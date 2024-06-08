from typing import *
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class PersistenceModel:

    class PersistenceModelType(Enum):
        THEORETICAL = "theory"
        THEORY_PLUS_COMPUTATIONAL = "theory+comp"
        THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS = "theory+comp+hyp"

    persistence_model_type: PersistenceModelType

    def describe(self) -> str:
        if (
            self.persistence_model_type
            == PersistenceModel.PersistenceModelType.THEORETICAL
        ):
            return "Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.'"
        elif (
            self.persistence_model_type
            == PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL
        ):
            return """Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.
            
Computationally, in the context of HoloOrbits, the sub-constructs in the theoretical definition of 'persistence' is mapped to the following specific learner actions in the learning environment.
            
1. "keeping at a task" -> MEASURE, USE_CALCULATOR
2. "finishing it" -> SUBMIT
3. "despite the obstacles" -> ASK_FOR_HELP, ASK_FOR_HINT"""
        elif (
            self.persistence_model_type
            == PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS
        ):
            return """Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.
            
Computationally, in the context of HoloOrbits, the sub-constructs in the theoretical definition of 'persistence' is mapped to the following specific learner actions in the learning environment.

1. "keeping at a task" -> MEASURE, USE_CALCULATOR
2. "finishing it" -> SUBMIT
3. "despite the obstacles" -> ASK_FOR_HELP, ASK_FOR_HINT

Concretely, act according to the following hypothesis:
1. For any given value of TIME_ELAPSED in the state, learners with higher persistence levels are less likely to abandon the session compared to learners with lower persistence levels."""


@dataclass
class Learner:
    persistence_level: int
    geometry_proficiency: int
