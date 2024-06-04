from typing import *
from pprint import pprint
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Import OpenAI API
from openai import OpenAI
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv(".env.secret")


class PersistenceModelType(Enum):
    THEORETICAL = "theory"
    THEORY_PLUS_COMPUTATIONAL = "theory+comp"
    THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS = "theory+comp+hyp"


@dataclass
class GenerationSample:
    prompt_version: str
    persistence_model: str


class PromptFragment(ABC):

    @staticmethod
    @abstractmethod
    def describe() -> str:
        raise NotImplementedError


class LearningTask(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """Your goal in the learning task is to show that the motion of the orbiting planet in the planetary systems is in accordance with Kepler's First Law."""
    

class LearnerCharacteristics(PromptFragment):

    @staticmethod
    def describe() -> str:
        return """The learner is a high school student who has just learned about Kepler's First Law and is trying to apply it to the motion of the orbiting planet in the planetary system."""


class State(PromptFragment, ABC):

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
        return State.describe_variables()


class Action(PromptFragment):

    @staticmethod
    def describe() -> str:
        raise NotImplementedError


@dataclass
class HOStateA(State, PromptFragment):
    time_elapsed_in_minutes: int
    num_submission_attempts: int

    def str_format(self):
        return f"Time elapsed: {self.time_elapsed_in_minutes} minutes, Num submission attempts: {self.num_submission_attempts}"

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
class HOStateB(State, PromptFragment):
    time_elapsed_in_minutes: int
    num_failed_submission_attempts: int

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
class HOActionSpaceA(Action):

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
class HOActionSpaceB(Action):

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


def assemble_prompt(state: State, action_space: Type[Action]) -> str:
    pass


def generate_next_action(
    persistence_model: PersistenceModelType,
    learner_characteristics: LearnerCharacteristics,
    state: State,
    action_space: Type[Enum],
) -> Action:
    """Assemble all components of the prompt and call the LLM."""
    prompt = assemble_prompt(
        persistence_model, learner_characteristics, state, action_space
    )
    gpt_response = gpt_call(prompt)
    if gpt_response in action_space:
        return Action(gpt_response)
    else:
        raise ValueError("Invalid action generated by the model.")
    return gpt_response


def run_simulations():
    pass


if __name__ == "__main__":
    valid_action_str = "SUBMIT"
    invalid_action_str = "INVALID"
    print(valid_action_str in HOActionSpaceA.actions())
    print(invalid_action_str in HOActionSpaceA.actions())

