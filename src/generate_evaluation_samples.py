from typing import *
from dataclasses import dataclass, field
from enum import Enum
import itertools
import random

random.seed(22)

from langsmith import Client
from dotenv import load_dotenv

load_dotenv(".env.secret")


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
class State:
    state_space_name: str

    def describe_state(self) -> str:
        raise NotImplementedError


@dataclass
class ActionSpace:

    action_space_name: str
    actions: Dict[Text, Text]

    def describe_action_space(self) -> str:
        return "\n".join([f"{action}: {desc}" for action, desc in self.actions.items()])


@dataclass
class HOStateA(State):

    state_space_name: str = "HOStateSpaceA"
    time_elapsed_in_minutes: int = 0
    num_submission_attempts: int = 0

    def describe_state(self) -> str:
        return f"""State:\nTIME_ELAPSED (the number of minutes that have passed since the start of the session): {self.time_elapsed_in_minutes} minutes\nNUM_SUBMISSION_ATTEMPTS (the number of times the user has submitted an answer since the start of the session): {self.num_submission_attempts}"""


@dataclass
class HOActionSpaceA(ActionSpace):

    action_space_name: str = "HOActionSpaceA"
    actions: Dict[Text, Text] = field(default_factory=lambda: {
        "SUBMIT": "The user submits their answer.",
        "MEASURE": "The user measures the distance between two points.",
        "USE_CALCULATOR": "The user uses the calculator.",
        "ASK_FOR_HELP": "The user asks for help.",
        "ASK_FOR_HINT": "The user asks for a hint.",
        "ASK_FOR_EXPLANATION": "The user asks for an explanation.",
        "ASK_FOR_SOLUTION": "The user asks for a solution.",
        "EXIT": "The user exits the session.",
    })


@dataclass
class Learner:
    persistence_level: int
    geometry_proficiency: int


@dataclass
class StateActionPair:
    """Corresponds a single row in a dataset."""

    dataset_split: Text  # "calibration" or "evaluation"
    learner: Learner
    persistence_model: PersistenceModel
    state: State
    action_space: ActionSpace

    @property
    def as_langsmith_sample(self):
        return {
            "inputs": {
                "persistence_level": self.learner.persistence_level,
                "geometry_proficiency": self.learner.geometry_proficiency,
                "state": self.state.describe_state(),
                "action_space": self.action_space.describe_action_space(),
                "persistence_definition": self.persistence_model.describe(),
            },
            "outputs": {},
            "metadata": {
                "split": self.dataset_split,
                "persistence_model_level": self.persistence_model.persistence_model_type.value,
                "state_space_name": self.state.state_space_name,
                "action_space_name": self.action_space.action_space_name,
                "persistence_level": self.learner.persistence_level,
                "geometry_proficiency": self.learner.geometry_proficiency,
                **self.state.__dict__,
            },
        }


def generate_calibration_samples():
    sweep_space = {
        "persistence_level": [1, 2, 3, 4, 5],
        "geometry_proficiency": [1, 2, 3, 4, 5],
        "persistence_model": [
            PersistenceModel.PersistenceModelType.THEORETICAL,
            PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL,
            PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS,
        ],
        "state": [HOStateA(time_elapsed_in_minutes=0, num_submission_attempts=0)],
        "action_space": [HOActionSpaceA()],
    }
    return [
        StateActionPair(
            dataset_split="calibration",
            learner=Learner(
                persistence_level=pl,
                geometry_proficiency=gp,
            ),
            persistence_model=PersistenceModel(pm),
            state=state,
            action_space=acs,
        )
        for pl, gp, pm, state, acs in itertools.product(*sweep_space.values())
    ]


def generate_evaluation_samples():
    raise NotImplementedError


if __name__ == "__main__":
    client = Client()
    dataset = client.create_dataset("persistsim-sweep-2")
    for sample in generate_calibration_samples():
        langsmith_sample = sample.as_langsmith_sample
        client.create_example(
            inputs=langsmith_sample["inputs"],
            outputs=langsmith_sample["outputs"],
            metadata=langsmith_sample["metadata"],
            dataset_id=dataset.id,
        )
