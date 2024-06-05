from typing import *
from pprint import pprint
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import os

from fire import Fire
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum

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


class LearningTask(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """Your goal in the learning task is to show that the motion of the orbiting planet in the planetary systems is in accordance with Kepler's First Law."""


class Learner(PromptFragment):

    persistence_level: int

    @staticmethod
    def describe() -> str:
        return """The learner is a high school student who has just learned about Kepler's First Law and is trying to apply it to the motion of the orbiting planet in the planetary system."""
    
    def str_format(self) -> str:
        return f"""Persistence Level: {self.persistence_level}"""


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
    def describe() -> str:
        raise NotImplementedError


@dataclass
class HOStateA(StateSpace):

    state_space_name: str = "A"
    time_elapsed_in_minutes: int = 0
    num_submission_attempts: int = 0

    def str_format(self):
        return f"Time elapsed: {self.time_elapsed_in_minutes} minutes, Num submission attempts: {self.num_submission_attempts}"
    
    @staticmethod
    def describe() -> str:
        return HOStateA.describe_variables()

    @staticmethod
    def describe_variables() -> str:
        return """
        time_elapsed_in_minutes represents the number of minutes that have passed since the start of the session.
        num_submission_attempts represents the number of times the user has attempted to submit their answer since the start of the session.
        """

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
        return """
        time_elapsed_in_minutes represents the number of minutes that have passed since the start of the session.
        num_failed_submission_attempts represents the number of times the user has submitted an incorrect answer since the start of the session.
        """

    def str_format(self):
        return f"Time elapsed: {self.time_elapsed_in_minutes} minutes, Num failed submission attempts: {self.num_failed_submission_attempts}"

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
        return """
        SUBMIT: The user submits their answer.
        MEASURE: The user measures the distance between two points.
        USE_CALCULATOR: The user uses the calculator.
        ASK_FOR_HELP: The user asks for help.
        ASK_FOR_HINT: The user asks for a hint.
        ASK_FOR_EXPLANATION: The user asks for an explanation.
        ASK_FOR_SOLUTION: The user asks for a solution.
        EXIT: The user exits the session.
        """


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
        return """
        ATTEMPT_SUBMISSION: The user submits their answer.
        MEASURE_DISTANCE: The user measures the distance between two points.
        USE_CALCULATOR: The user uses the calculator.
        ABANDON_SESSION: The user abandons the session.
        """


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


def gpt_call(prompt: str, temperature: int = 2, model: str = "gpt-4"):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
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
        persistence_model.value
        + "\n"
        + learner_characteristics.describe()
        + "\n"
        + state.str_format()
        + "\n"
        + action_space.describe()
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
    gpt_response = gpt_call(prompt)
    if gpt_response in action_space:
        return ActionSpace(gpt_response)
    else:
        raise ValueError("Invalid action generated by the model.")


def generate_input_combinations():
    from itertools import product

    persistence_models = [
        PersistenceModelType.THEORETICAL,
        PersistenceModelType.THEORY_PLUS_COMPUTATIONAL,
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
