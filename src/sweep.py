from typing import *
from dataclasses import dataclass, field
from functools import partial
import itertools
import random
import os

random.seed(22)

from langsmith import Client, traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain import hub
from langsmith.wrappers import wrap_openai
from langsmith.evaluation import evaluate, evaluate_existing
from langsmith.schemas import Example, Run
from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env.secret")
load_dotenv(".env")

from action_spaces import ActionSpace, HOActionSpaceB
from state_spaces import State, HOStateA
from learners import (
    Learner,
    LearnerCharacteristicModel,
    ModelType,
    create_geometry_proficiency_model,
    create_persistence_model,
)


@dataclass
class Vignette:
    """Corresponds a single row in a dataset (akin to a vignette used in psychological research)."""

    sweep_id: Text
    dataset_split: Text  # "validation" or "test"
    learner: Learner
    model_type: ModelType
    state: State
    action_space: ActionSpace

    @property
    def as_langsmith_sample(self):
        return {
            # For the prompt template
            "inputs": {
                "persistence_level": self.learner.persistence_level,
                "geometry_proficiency_level": self.learner.geometry_proficiency_level,
                "persistence_model": self.learner.persistence_model.describe(),
                "geometry_proficiency_model": self.learner.geometry_proficiency_model.describe(),
                "state": self.state.describe_state(),
                "action_space": self.action_space.describe_action_space(),
            },
            "outputs": {},
            # For filtering
            "metadata": {
                "sweep_id": self.sweep_id,
                "split": self.dataset_split,
                "model_type": self.model_type,
                "state_space_name": self.state.state_space_name,
                "action_space_name": self.action_space.action_space_name,
                "persistence_level": self.learner.persistence_level,
                "geometry_proficiency_level": self.learner.geometry_proficiency_level,
                **self.state.__dict__,
            },
        }


@dataclass
class Sweep:
    sweep_id: Text

    dataset_name: Text
    dataset_split: Text

    model_name: Text
    temperature: float
    prompt_name: Text

    persistence_levels: List[int]
    geometry_proficiency_levels: List[int]

    model_types: ModelType

    states: List[State]
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
            "persistence_levels": self.persistence_levels,
            "geometry_proficiency_levels": self.geometry_proficiency_levels,
            "model_types": self.model_types,
            "states": self.states,
            "action_spaces": [self.action_space],
        }

    def _predict_over_dataset(
        self,
        dataset: Text,
        dataset_filters: dict,
        num_generations_per_sample: int = 1,
        experiment_prefix: Text = None,
    ):
        """Predict over the dataset and log the results."""

        def extract_action_label(
            action_space: "ActionSpace", message: "AIMessage"
        ) -> Text:
            for action in action_space.actions.keys():
                base_action = action.split("(")[
                    0
                ]  # Get the base action before any parameters
                if base_action in message.content:
                    return action
            return "UNPREDICTED"

        # Run evaluation for the first time
        chain = (
            self.prompt
            | self.chat_model
            | partial(extract_action_label, self.action_space)
        )
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
                if run.outputs is None:
                    continue
                if run.outputs["output"] == action_label:
                    print(run.outputs["output"])
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
            ]
            + [
                HOActionSpaceB.productive_measurement_percentage,
                HOActionSpaceB.unproductive_measurement_percentage,
            ],
        )
        return {
            f"action_{result.key}": result.score
            for result in results._summary_results["results"]
        }

    def _log_next_action_distribution(
        self, existing_experiment_id: Text, action_space: ActionSpace
    ):
        """Log the distribution of the next action in the existing experiment to a CSV."""

        log_row = {
            "experiment_id": existing_experiment_id,
            "action_space": action_space.action_space_name,
            "split": self.dataset_split,
            "persistence_levels": self.persistence_levels,
            "geometry_proficiency_levels": self.geometry_proficiency_levels,
            "model_types": [t.value for t in self.model_types],
            "model": self.model_name,
            # **{
            #     f"state_{key.lower()}": value
            #     for key, value in state.state_variables.items()
            # },
            **self._get_next_action_distribution(existing_experiment_id, action_space),
        }
        # Read the list of distributions from a CSV onto a Pandas DataFrame, insert the new distribution, and write back to the CSV
        if os.path.exists(f"results/action_distribution.csv"):
            try:
                existing = pd.read_csv(f"results/action_distribution.csv")
                existing = pd.concat([existing, pd.DataFrame([log_row])])
                existing.to_csv(f"results/action_distribution.csv", index=False)
            except Exception as e:
                print(f"Exception: {e}")
                df = pd.DataFrame([log_row])
                df.to_csv(f"results/action_distribution.csv", index=False)
        else:
            df = pd.DataFrame([log_row])
            df.to_csv(f"results/action_distribution.csv", index=False)
        print(f"Logged next action distribution to results/action_distribution.csv")

    def _create_evaluation_dataset(self):
        """Create a LangSmith dataset with all possible combinations of the sweep parameters."""
        client = Client()
        try:
            dataset = client.create_dataset(self.dataset_name)
        except Exception as e:
            print(f"Exception: {e}")
            dataset = client.read_dataset(dataset_name=self.dataset_name)
        samples = [
            Vignette(
                sweep_id=self.sweep_id,
                dataset_split=self.dataset_split,
                learner=Learner(
                    persistence_level=persistence_level,
                    geometry_proficiency_level=geometry_proficiency_level,
                    persistence_model=create_persistence_model(model_type),
                    geometry_proficiency_model=create_geometry_proficiency_model(
                        model_type
                    ),
                ),
                model_type=model_type,
                state=state,
                action_space=action_space,
            )
            for persistence_level, geometry_proficiency_level, model_type, state, action_space in itertools.product(
                *self.sweep_dict.values()
            )
        ]
        for sample in samples:
            langsmith_sample = sample.as_langsmith_sample
            client.create_example(
                inputs=langsmith_sample["inputs"],
                outputs=langsmith_sample["outputs"],
                metadata=langsmith_sample["metadata"],
                dataset_id=dataset.id,
            )

    def run(self, num_generations_per_sample: int = 1):
        """Create, predict over, and log the next action distribution for the evaluation dataset."""
        self._create_evaluation_dataset()
        experiment_name = self._predict_over_dataset(
            self.dataset_name,
            {"split": self.dataset_split, "sweep_id": self.sweep_id},
            experiment_prefix="experiment-",
            num_generations_per_sample=num_generations_per_sample,
        ).experiment_name
        self._log_next_action_distribution(experiment_name, self.action_space)
