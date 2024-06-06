import json
from typing import *
from pprint import pprint
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import os
import itertools

from fire import Fire
from openai import OpenAI
from dotenv import load_dotenv
import randomname

load_dotenv(".env.secret")


class PromptFragment(ABC):

    @staticmethod
    @abstractmethod
    def describe() -> str:
        raise NotImplementedError

    @staticmethod
    def describe_with_params(**kwargs) -> str:
        raise NotImplementedError


class PersistenceModel(PromptFragment):

    class PersistenceModelType(Enum):
        THEORETICAL = "theory"
        THEORY_PLUS_COMPUTATIONAL = "theory+comp"
        THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS = "theory+comp+hyp"

    @staticmethod
    def describe_with_params(**kwargs) -> str:
        assert "persistence_model_type" in kwargs and isinstance(
            kwargs["persistence_model_type"], PersistenceModel.PersistenceModelType
        )

        if (
            kwargs["persistence_model_type"]
            == PersistenceModel.PersistenceModelType.THEORETICAL
        ):
            return "Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.'"
        elif (
            kwargs["persistence_model_type"]
            == PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL
        ):
            return """Theoretically, 'persistence' is defined as 'Keeping at a task and finishing it despite the obstacles (such as opposition or discouragement) or the effort involved.
            
Computationally, in the context of HoloOrbits, the sub-constructs in the theoretical definition of 'persistence' is mapped to the following specific learner actions in the learning environment.
            
1. "keeping at a task" -> MEASURE, USE_CALCULATOR
2. "finishing it" -> SUBMIT
3. "despite the obstacles" -> ASK_FOR_HELP, ASK_FOR_HINT"""
        elif (
            kwargs["persistence_model_type"]
            == PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS
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
        raise NotImplementedError


class DefaultSystemPrompt(SystemPrompt):
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


@dataclass
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
    def get_sweep_values() -> Dict[Text, List[Any]]:
        raise NotImplementedError

    @staticmethod
    def get_sweep(sweep_values: Dict[Text, List[Any]]) -> Iterator[Dict[str, Any]]:
        keys = sweep_values.keys()
        values = itertools.product(*sweep_values.values())
        for value in values:
            yield dict(zip(keys, value))

    @staticmethod
    @abstractmethod
    def describe_state(state: Dict[Text, Any]) -> str:
        raise NotImplementedError

    @staticmethod
    def describe():
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
class HOStateSpaceA(StateSpace):

    state_space_name: str = "A"

    @staticmethod
    def describe_state(state: Dict[Text, Any]):
        return f"State Space:\nTIME_ELAPSED: {state['time_elapsed_in_minutes']} minutes, NUM_SUBMISSION_ATTEMPTS: {state['num_submission_attempts']}"

    @staticmethod
    def describe() -> str:
        return """TIME_ELAPSED represents the number of minutes that have passed since the start of the session.
NUM_SUBMISSION_ATTEMPTS represents the number of times the user has attempted to submit their answer since the start of the session."""

    @staticmethod
    def get_sweep_values() -> Dict[str, List[int]]:
        return {
            "time_elapsed_in_minutes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "num_submission_attempts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }


@dataclass
class HOStateSpaceB(StateSpace):

    state_space_name: str = "B"
    time_elapsed_in_minutes: int = 0
    num_failed_submission_attempts: int = 0

    @staticmethod
    def describe() -> str:
        return """TIME_ELAPSED_IN_MINUTES represents the number of minutes that have passed since the start of the session.
NUM_FAILED_SUBMISSION_ATTEMPTS represents the number of times the user has submitted an incorrect answer since the start of the session."""

    @staticmethod
    def describe_state(state: Dict[Text, Any]):
        return f"State Space:\nTIME_ELAPSED: {state['time_elapsed_in_minutes']} minutes, NUM_FAILED_SUBMISSION_ATTEMPTS: {state['num_failed_submission_attempts']}"

    @staticmethod
    def get_sweep_values() -> Dict[str, Any]:
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
    """Corresponds a single row in a dataset."""

    experiment_id: Text
    system_prompt: Type[SystemPrompt]
    persistence_model_type: PersistenceModel.PersistenceModelType
    learner: Learner
    state_space: Type[StateSpace]
    action_space: Type[ActionSpace]
    state: Dict[Text, Any]

    def assemble_prompt(self) -> str:
        return (
            PersistenceModel.describe_with_params(
                persistence_model_type=self.persistence_model_type
            )
            + "\n"
            + self.learner.describe()
            + "\n"
            + self.state_space.describe()
            + "\n"
            + self.state_space.describe_state(self.state)
            + "\n"
            + self.action_space.describe()
            + "\n"
            + SimulatorInstruction.describe()
        )

    def generate_next_action(self) -> ActionSpace:
        """Assemble all components of the prompt and call the LLM."""
        prompt = self.assemble_prompt()
        pprint(prompt)
        gpt_response = gpt_call(prompt)
        pprint(gpt_response)
        if gpt_response in self.action_space.actions():
            return gpt_response
        else:
            raise ValueError("Invalid action generated by the model.")

    def as_openai_request(self, model: str = "gpt-3.5-turbo"):
        return {
            "custom_id": self.experiment_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.system_prompt.describe()},
                    {"role": "user", "content": self.assemble_prompt()},
                ],
            },
        }


def gpt_call(
    client: OpenAI, prompt: str, temperature: int = 1.5, model: str = "gpt-3.5-turbo"
):
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


def generate_samples(
    experiment_id: str,
    persistence_model_types: List[PersistenceModel.PersistenceModelType],
    learners: List[Learner],
    state_spaces: List[StateSpace],
    action_spaces: List[Type[Enum]],
    model: str = "gpt-3.5-turbo",
):
    generation_samples = []
    for pmt, ss, acs, lc in itertools.product(
        persistence_model_types, state_spaces, action_spaces, learners
    ):
        for state in ss.get_sweep(ss.get_sweep_values()):
            generation_samples.append(
                GenerationSample(
                    experiment_id=experiment_id,
                    system_prompt=DefaultSystemPrompt,
                    persistence_model_type=pmt,
                    learner=lc,
                    state_space=ss,
                    action_space=acs,
                    state=state,
                )
            )
    return generation_samples


def run_simulations(client: OpenAI):
    experiment_id = randomname.get_name()
    generation_samples = generate_samples(
        experiment_id=experiment_id,
        persistence_model_types=[
            PersistenceModel.PersistenceModelType.THEORY_PLUS_COMPUTATIONAL_PLUS_HYPOTHESIS
        ],
        learners=[Learner(persistence_level=5, geometry_proficiency=5)],
        state_spaces=[HOStateSpaceA, HOStateSpaceB],
        action_spaces=[HOActionSpaceA, HOActionSpaceB],
    )
    generation_requests = [sample.as_openai_request() for sample in generation_samples]
    # Save requests to JSONL file to a temporary file in openai_batch_requests/
    with open(f"runs/requests/{experiment_id}.jsonl", "w") as f:
        for request in generation_requests:
            f.write(json.dumps(request) + "\n")
    # Call OpenAI batch prediction API
    file_creation_response = client.files.create(
        file=open(f"runs/requests/{experiment_id}.jsonl", "rb"), purpose="batch"
    )
    breakpoint()
    file_id = file_creation_response["id"]
    responses = client.batch.create(generation_requests)
    # Save responses to JSONL file
    with open(f"runs/responses/{experiment_id}.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    # Create dir if not exists

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    run_simulations(client)
