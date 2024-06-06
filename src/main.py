from typing import *
from pprint import pprint
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import os

from fire import Fire
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.secret")


class PersistenceModelType(Enum):
    THEORETICAL = "theory"
    THEORY_PLUS_COMPUTATIONAL = "theory+comp"
    THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS = "theory+comp+hyp"


class PromptFragment(ABC):

    @staticmethod
    @abstractmethod
    def describe() -> str:
        raise NotImplementedError

    @staticmethod
    def describe_with_params(**kwargs) -> str:
        raise NotImplementedError


class PersistenceModel(PromptFragment):
    @staticmethod
    def describe_with_params(**kwargs) -> str:
        assert "persistence_model_type" in kwargs and isinstance(
            kwargs["persistence_model_type"], PersistenceModelType
        )

        if kwargs["persistence_model_type"] == PersistenceModelType.THEORETICAL:
            return "Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.'"
        elif (
            kwargs["persistence_model_type"]
            == PersistenceModelType.THEORY_PLUS_COMPUTATIONAL
        ):
            return """Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.
            
Computationally, in the context of HoloOrbits, the sub-constructs in the theoretical definition of 'persistence' is mapped to the following specific learner actions in the learning environment.
            
1. "keeping at a task" -> MEASURE, USE_CALCULATOR
2. "finishing it" -> SUBMIT
3. "despite the obstacles" -> ASK_FOR_HELP, ASK_FOR_HINT"""
        elif (
            kwargs["persistence_model_type"]
            == PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS
        ):
            return """Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.
            
Computationally, in the context of HoloOrbits, the sub-constructs in the theoretical definition of 'persistence' is mapped to the following specific learner actions in the learning environment.

1. "keeping at a task" -> MEASURE, USE_CALCULATOR
2. "finishing it" -> SUBMIT
3. "despite the obstacles" -> ASK_FOR_HELP, ASK_FOR_HINT

Concretely, act according to the following hypothesis:
1. For any given value of TIME_ELAPSED in the state, learners with higher persistence levels are less likely to abandon the session compared to learners with lower persistence levels."""


class SystemPrompt(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """You are a simulated learner agent working in a learning environment designed to test your understanding of Kepler's First Law. Given a scenario in the learning environment, you will generate the next action that a 13 year old human learner who possesses the given learner characteristics would most likely perform in the given situation. The stipulated class period for this activity is 40 minutes. The teacher has instructed you to work on the activity for the entire class period."""


class LearningTask(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """Your goal in the learning task is to show that the motion of the orbiting planet in the planetary systems is in accordance with Kepler's First Law."""


class SimulatorInstruction(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """Given the scenario, choose the most likely next action such that it is consistent with your Learner Characteristics. 

Next Action (one of the possible actions defined the action space):"""


class Learner(PromptFragment):

    persistence_level: int
    geometry_proficiency: int

    def __post_init__(self):
        if self.persistence_level not in range(1, 11):
            raise ValueError("Persistence level must be between 1 and 10.")
        if self.geometry_proficiency not in range(1, 11):
            raise ValueError("Geometry proficiency must be between 1 and 10.")

    @staticmethod
    def describe() -> str:
        return """The learner is a high school student who has just learned about Kepler's First Law and is trying to apply it to the motion of the orbiting planet in the planetary system."""

    def str_format(self) -> str:
        return f"""Persistence Level: {self.persistence_level}
        Geometry Proficiency: {self.geometry_proficiency}"""


@dataclass
class StateSpace(PromptFragment):

    state_space_name: str

    @staticmethod
    @abstractmethod
    def get_sweep() -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def str_format(self) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def describe_variables() -> str:
        raise NotImplementedError

    @staticmethod
    def desribe():
        return StateSpace.describe_variables()


@dataclass
class ActionSpace(PromptFragment):

    action_space_name: str

    @staticmethod
    def actions():
        raise NotImplementedError

    @staticmethod
    def describe() -> str:
        raise NotImplementedError


@dataclass
class HOStateA(StateSpace):

    state_space_name: str = "A"
    time_elapsed_in_minutes: int = 0
    num_submission_attempts: int = 0

    def str_format(self):
        return f"State Space:\nTIME_ELAPSED: {self.time_elapsed_in_minutes} minutes, NUM_SUBMISSION_ATTEMPTS: {self.num_submission_attempts}"

    @staticmethod
    def describe() -> str:
        return HOStateA.describe_variables()

    @staticmethod
    def describe_variables() -> str:
        return """TIME_ELAPSED represents the number of minutes that have passed since the start of the session.
NUM_SUBMISSION_ATTEMPTS represents the number of times the user has attempted to submit their answer since the start of the session."""

    @staticmethod
    def get_sweep() -> Dict[str, int]:
        return {
            "time_elapsed_in_minutes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "num_submission_attempts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }


@dataclass
class HOStateB(StateSpace):

    state_space_name: str = "B"
    time_elapsed_in_minutes: int = 0
    num_failed_submission_attempts: int = 0

    @staticmethod
    def describe() -> str:
        return HOStateB.describe_variables()

    @staticmethod
    def describe_variables():
        return """TIME_ELAPSED_IN_MINUTES represents the number of minutes that have passed since the start of the session.
NUM_FAILED_SUBMISSION_ATTEMPTS represents the number of times the user has submitted an incorrect answer since the start of the session."""

    def str_format(self):
        return f"State Space:\nTIME_ELAPSED: {self.time_elapsed_in_minutes} minutes, NUM_FAILED_SUBMISSION_ATTEMPTS: {self.num_failed_submission_attempts}"

    @staticmethod
    def get_sweep() -> Dict[str, int]:
        return {
            "time_elapsed_in_minutes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "num_failed_submission_attempts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }


@dataclass
class HOActionSpaceA(ActionSpace):

    action_space_name: str = "A"

    @staticmethod
    def actions():
        return {
            "SUBMIT": "The user submits their answer.",
            "MEASURE": "The user measures the distance between two points.",
            "USE_CALCULATOR": "The user uses the calculator.",
            "ASK_FOR_HELP": "The user asks for help.",
            "ASK_FOR_HINT": "The user asks for a hint.",
            "ASK_FOR_EXPLANATION": "The user asks for an explanation.",
            "ASK_FOR_SOLUTION": "The user asks for a solution.",
            "EXIT": "The user exits the session.",
        }

    @staticmethod
    def describe() -> str:
        return """Action Space:\nSUBMIT: The user submits their answer.
MEASURE: The user measures the distance between two points.
USE_CALCULATOR: The user uses the calculator.
ASK_FOR_HELP: The user asks for help.
ASK_FOR_HINT: The user asks for a hint.
ASK_FOR_EXPLANATION: The user asks for an explanation.
ASK_FOR_SOLUTION: The user asks for a solution.
EXIT: The user exits the session."""


@dataclass
class HOActionSpaceB(ActionSpace):

    action_space_name: str = "B"

    @staticmethod
    def actions():
        return {
            "ATTEMPT_SUBMISSION": "The user submits their answer.",
            "MEASURE_DISTANCE": "The user measures the distance between two points.",
            "USE_CALCULATOR": "The user uses the calculator.",
            "ABANDON_SESSION": "The user abandons the session.",
        }

    @staticmethod
    def describe() -> str:
        return """Action Space:\nATTEMPT_SUBMISSION: The user submits their answer.
MEASURE_DISTANCE: The user measures the distance between two points.
USE_CALCULATOR: The user uses the calculator.
ABANDON_SESSION: The user abandons the session."""


@dataclass
class GenerationSample:
    prompt_version: str
    persistence_model: PersistenceModelType
    learner_characteristics: Learner
    state: StateSpace
    action_space: Type[Enum]
    generated_action: ActionSpace


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def gpt_call(prompt: str, temperature: int = 1.5, model: str = "gpt-3.5-turbo"):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SystemPrompt.describe()},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        temperature=temperature,
    )
    return response.choices[0].message.content


def assemble_prompt(
    persistence_model: PersistenceModelType,
    learner_characteristics: Learner,
    state: StateSpace,
    action_space: Type[Enum],
) -> str:
    return (
        PersistenceModel.describe_with_params(persistence_model_type=persistence_model)
        + "\n"
        + learner_characteristics.describe()
        + "\n"
        + state.str_format()
        + "\n"
        + action_space.describe()
        + "\n"
        + SimulatorInstruction.describe()
    )


def generate_next_action(
    persistence_model: PersistenceModelType,
    learner_characteristics: Learner,
    state: StateSpace,
    action_space: Type[Enum],
) -> ActionSpace:
    """Assemble all components of the prompt and call the LLM."""
    prompt = assemble_prompt(
        persistence_model, learner_characteristics, state, action_space
    )
    pprint(prompt)
    gpt_response = gpt_call(prompt)
    pprint(gpt_response)
    breakpoint()
    if gpt_response in action_space.actions():
        return ActionSpace(gpt_response)
    else:
        raise ValueError("Invalid action generated by the model.")


def generate_input_combinations():
    from itertools import product

    persistence_models = [
        # PersistenceModelType.THEORETICAL,
        # PersistenceModelType.THEORY_PLUS_COMPUTATIONAL,
        PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS,
    ]
    state_spaces = [HOStateA, HOStateB]
    action_spaces = [HOActionSpaceA, HOActionSpaceB]
    learner_characteristics = [Learner]

    for pm, sc, ac, lc in product(
        persistence_models, state_spaces, action_spaces, learner_characteristics
    ):
        print(f"Generating samples for {pm}, {sc}, {ac}, {lc}")
        state_space = sc.get_sweep()
        for state in product(*state_space.values()):
            print(f"Generating sample for state {state}")
            yield GenerationSample(
                prompt_version="v1",
                persistence_model=pm,
                learner_characteristics=lc,
                state=sc(*state),
                action_space=ac,
                generated_action=generate_next_action(pm, lc, sc(*state), ac),
            )


def run_simulations():
    pass


if __name__ == "__main__":
    input_combo = next(generate_input_combinations())
    print(input_combo)
