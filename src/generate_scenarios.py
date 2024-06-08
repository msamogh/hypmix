from typing import *
from dataclasses import dataclass, field
from enum import Enum
import itertools
import random

random.seed(22)

from langsmith import Client
from dotenv import load_dotenv

load_dotenv(".env.secret")

from state_spaces import State, HOStateA
from action_spaces import ActionSpace, HOActionSpaceA
from learners import Learner, PersistenceModel


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

def generate_samples(split, sweep_space):
    """Take a sweep space and generate all possible combinations of the parameters for a dataset split."""
    return [
        StateActionPair(
            dataset_split=split,
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

SWEEP_SPACE_FOR_CALIBRATION_SPLIT = {
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

SWEEP_SPACE_FOR_EVALUATION_SPLIT = {
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

if __name__ == "__main__":
    client = Client()
    dataset = client.create_dataset("persistsim-sweep-2")
    for sample in generate_samples("calibration", SWEEP_SPACE_FOR_CALIBRATION_SPLIT):
        langsmith_sample = sample.as_langsmith_sample
        client.create_example(
            inputs=langsmith_sample["inputs"],
            outputs=langsmith_sample["outputs"],
            metadata=langsmith_sample["metadata"],
            dataset_id=dataset.id,
        )
