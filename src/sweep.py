from typing import *
from dataclasses import dataclass, field
from functools import partial
import itertools
import random
import os

random.seed(22)

from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from langsmith.evaluation import evaluate, evaluate_existing
from langsmith.schemas import Example, Run
from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env.secret")

from action_spaces import ActionSpace, HOActionSpaceA
from state_spaces import State, HOStateA
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


@dataclass
class Sweep:
    dataset_name: Text
    dataset_split: Text

    model_name: Text
    temperature: float
    prompt_name: Text

    persistence_level: List[int]
    geometry_proficiency: List[int]
    persistence_model: List[PersistenceModel.PersistenceModelType]
    state: List[State]
    action_space: ActionSpace

    def __post_init__(self):
        # Initialize client
        self.client = Client()

        # Load components
        self.prompt = hub.pull(self.prompt_name)
        self.chat_model = ChatOpenAI(
            model=self.model_name, temperature=self.temperature
        )
        self.output_parser = StrOutputParser()

    @property
    def sweep_dict(self):
        """Used to populate the metadata of the LangSmith dataset."""
        return {
            "persistence_level": self.persistence_level,
            "geometry_proficiency": self.geometry_proficiency,
            "persistence_model": self.persistence_model,
            "state": self.state,
            "action_space": self.action_space,
        }

    def _predict_over_dataset(
        self,
        dataset: Text,
        dataset_filters: dict,
        num_generations_per_sample: int = 1,
        experiment_prefix: Text = None,
    ):
        """Predict over the dataset and log the results."""
        # Run evaluation for the first time
        chain = self.prompt | self.chat_model | self.output_parser
        results = evaluate(
            chain.invoke,
            data=self.client.list_examples(
                dataset_name=dataset, metadata=dataset_filters
            ),
            experiment_prefix=experiment_prefix,
            num_repetitions=num_generations_per_sample,
        )
        return results

    def _get_next_action_distribution(
        self, existing_experiment_id: Text, action_space: ActionSpace
    ):
        """Get the distribution of the next action in the existing experiment."""

        def percentage_of_action(
            runs: list[Run], examples: list[Example], action_label: Text
        ) -> dict:
            """Calculate the percentage of a particular action in the list of runs."""
            action_count = 0
            for i, run in enumerate(runs):
                if run.outputs["output"] == action_label:
                    action_count += 1
            return {
                "key": f"{action_label.lower()}_percentage",
                "score": action_count / len(runs),
            }

        results = evaluate_existing(
            existing_experiment_id,
            summary_evaluators=[
                partial(percentage_of_action, action_label=action_label)
                for action_label in action_space.actions.keys()
            ],
        )
        return {
            result.key: result.score for result in results._summary_results["results"]
        }

    def _log_next_action_distribution(
        self, existing_experiment_id: Text, action_space: ActionSpace
    ):
        """Log the distribution of the next action in the existing experiment to a CSV."""

        next_action_distribution = {
            "experiment_id": existing_experiment_id,
            **self._get_next_action_distribution(existing_experiment_id, action_space),
        }
        # Read the list of distributions from a CSV onto a Pandas DataFrame, insert the new distribution, and write back to the CSV
        if os.path.exists("results/action_distribution.csv"):
            try:
                existing = pd.read_csv("results/action_distribution.csv")
                existing = pd.concat(
                    [existing, pd.DataFrame([next_action_distribution])]
                )
                existing.to_csv("results/action_distribution.csv", index=False)
            except Exception as e:
                print(f"Exception: {e}")
                df = pd.DataFrame([next_action_distribution])
                df.to_csv("results/action_distribution.csv", index=False)
        else:
            df = pd.DataFrame([next_action_distribution])
            df.to_csv("results/action_distribution.csv", index=False)
        print("Logged next action distribution to results/action_distribution.csv")

    def _create_evaluation_dataset(self):
        """Create a LangSmith dataset with all possible combinations of the sweep parameters."""
        client = Client()
        dataset = client.create_dataset(self.dataset_name)
        samples = [
            StateActionPair(
                dataset_split=self.dataset_split,
                learner=Learner(
                    persistence_level=pl,
                    geometry_proficiency=gp,
                ),
                persistence_model=PersistenceModel(pm),
                state=state,
                action_space=acs,
            )
            for pl, gp, pm, state, acs in itertools.product(*self.sweep_dict.values())
        ]
        for sample in samples:
            langsmith_sample = sample.as_langsmith_sample
            client.create_example(
                inputs=langsmith_sample["inputs"],
                outputs=langsmith_sample["outputs"],
                metadata=langsmith_sample["metadata"],
                dataset_id=dataset.id,
            )

    def run(self, create_dataset: bool = False):
        """Create, predict over, and log the next action distribution for the evaluation dataset."""
        if create_dataset:
            self._create_evaluation_dataset()
        experiment_name = self._predict_over_dataset(
            self.dataset_name,
            {"split": self.dataset_split, "num_submission_attempts": 0},
        ).experiment_name
        self._log_next_action_distribution(experiment_name, self.action_space)
