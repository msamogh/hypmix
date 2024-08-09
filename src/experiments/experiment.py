import itertools
import os
import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import *

random.seed(22)

import pandas as pd
from dotenv import load_dotenv
from langchain import hub
from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate, evaluate_existing
from langsmith.schemas import Example, Run

load_dotenv(".env.secret")
load_dotenv(".env")

from environment.action_spaces import ActionSpace
from environment.state_spaces import State, StateSweep
from learner.learners import Learner, LearnerCharacteristicModel


@dataclass
class Vignette:
    """Corresponds a single row in a dataset (akin to a vignette used in psychological research)."""

    experiment_id: Text
    learner: Learner
    state: State
    state_sweep_name: Text

    @property
    def as_langsmith_sample(self):
        return {
            # For the prompt template
            "inputs": {
                "persistence_level": self.learner.persistence_level,
                "persistence_level_str": (
                    f"Your persistence is {self.learner.persistence_level}/10"
                    if self.learner.persistence_model
                    else ""
                ),
                "geometry_proficiency_level": self.learner.geometry_proficiency_level,
                "geometry_proficiency_level_str": (
                    f"Your geometry proficiency is {self.learner.geometry_proficiency_level}/10"
                    if self.learner.geometry_proficiency_level
                    else ""
                ),
                "persistence_model": (
                    self.learner.persistence_model.describe()
                    if self.learner.persistence_model
                    else ""
                ),
                "geometry_proficiency_model": (
                    self.learner.geometry_proficiency_model.describe()
                    if self.learner.geometry_proficiency_model
                    else ""
                ),
                "state": self.state.describe_state(),
                "state_sweep_name": self.state_sweep_name,
                "action_space": self.learner.action_space.describe_action_space(),
            },
            "outputs": {},
            # For filtering
            "metadata": {
                "experiment_id": self.experiment_id,
                "state_space_name": self.state.state_space_name,
                "persistence_model": str(self.learner.persistence_model),
                "geometry_proficiency_model": str(
                    self.learner.geometry_proficiency_model
                ),
                "state_sweep_name": self.state_sweep_name,
                "action_space_name": self.learner.action_space.action_space_name,
                "persistence_level": self.learner.persistence_level,
                "geometry_proficiency_level": self.learner.geometry_proficiency_level,
                **self.state.__dict__,
            },
        }


@dataclass
class Experiment:
    experiment_id: Text
    dataset_name: Text

    model_name: Text
    temperature: float
    prompt_name: Text

    persistence_model: LearnerCharacteristicModel
    geometry_proficiency_model: LearnerCharacteristicModel

    persistence_levels: List[int]
    geometry_proficiency_levels: List[int]

    state_sweep: StateSweep
    action_space: ActionSpace

    def __post_init__(self):
        # Initialize client
        self.client = Client()
        # Verify that all model types are the same
        # Load components
        self.prompt = hub.pull(self.prompt_name)
        self.chat_model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            model_kwargs={"seed": 420, "top_p": 0.01},
        )

    @property
    def experiment_dict(self):
        """Used to populate the metadata of the LangSmith dataset."""
        return {
            "persistence_model": [self.persistence_model],
            "geometry_proficiency_model": [self.geometry_proficiency_model],
            "persistence_levels": (
                self.persistence_levels if self.persistence_levels else [None]
            ),
            "geometry_proficiency_levels": (
                self.geometry_proficiency_levels
                if self.geometry_proficiency_levels
                else [None]
            ),
            "states": self.state_sweep.states,
            "state_sweep_name": [self.state_sweep.state_space_name],
            "action_spaces": [self.action_space],
        }

    def _predict_over_dataset(
        self,
        dataset: Text,
        dataset_filters: dict,
        num_generations_per_sample: int = 1,
        experiment_prefix: Text = None,
        fake_llm: bool = False,
    ):
        """Predict over the dataset and log the results."""

        def extract_action_label(
            action_space: "ActionSpace", message: "AIMessage"
        ) -> Text:
            for action in action_space.actions.keys():
                if isinstance(message, str):
                    if action in message:
                        return action
                elif action in message.content:
                    return action
            return "UNPREDICTED"

        # Run evaluation for the first time
        if fake_llm:
            chain = (
                self.prompt
                | FakeListLLM(responses=["MEASURE-A-P", "MEASURE-F1-F2"])
                | partial(extract_action_label, self.action_space)
            )
        else:
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

    def _calculate_aggregate_metrics(
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
                    action_count += 1
            print(f"{action_label.lower()}_percentage = {action_count / len(runs)}")
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
                action_space.productive_measurement_percentage,
                action_space.unproductive_measurement_percentage,
            ],
        )

        return {
            result.key: result.score for result in results._summary_results["results"]
        }

    def _create_evaluation_dataset(self):
        """Create a LangSmith dataset with all possible combinations of the experiment parameters."""
        client = Client()
        try:
            dataset = client.create_dataset(self.dataset_name)
        except Exception as e:
            print(f"Using existing dataset: {self.dataset_name}")
            dataset = client.read_dataset(dataset_name=self.dataset_name)
        vignettes = [
            Vignette(
                experiment_id=self.experiment_id,
                learner=Learner(
                    action_space=action_space,
                    persistence_level=persistence_level,
                    geometry_proficiency_level=geometry_proficiency_level,
                    persistence_model=persistence_model,
                    geometry_proficiency_model=geometry_proficiency_model,
                ),
                state_sweep_name=state_sweep_name,
                state=state,
            )
            for persistence_model, geometry_proficiency_model, persistence_level, geometry_proficiency_level, state, state_sweep_name, action_space in itertools.product(
                *self.experiment_dict.values()
            )
        ]
        for vignette in vignettes:
            langsmith_sample = vignette.as_langsmith_sample
            client.create_example(
                inputs=langsmith_sample["inputs"],
                outputs=langsmith_sample["outputs"],
                metadata=langsmith_sample["metadata"],
                dataset_id=dataset.id,
            )

    def run(self, num_generations_per_sample: int = 1, fake_llm: bool = False) -> dict:
        """Create, predict over, and log the next action distribution for the evaluation dataset."""
        self._create_evaluation_dataset()
        experiment = self._predict_over_dataset(
            self.dataset_name,
            {"experiment_id": self.experiment_id},
            experiment_prefix="experiment-",
            num_generations_per_sample=num_generations_per_sample,
            fake_llm=fake_llm,
        )
        next_actions = [
            experiment._results[i]["run"].outputs["output"]
            for i in range(len(experiment._results))
        ]

        time.sleep(1)

        return {
            "experiment_id": experiment.experiment_name,
            "action_space": self.action_space.action_space_name,
            "persistence_levels": self.persistence_levels,
            "geometry_proficiency_levels": self.geometry_proficiency_levels,
            "persistence_model": str(self.persistence_model),
            "geometry_proficiency_model": str(self.geometry_proficiency_model),
            "model": self.model_name,
            "state_sweep_name": self.state_sweep.state_space_name,
            "next_actions": next_actions,
            **self._calculate_aggregate_metrics(
                experiment.experiment_name, self.action_space
            ),
        }
