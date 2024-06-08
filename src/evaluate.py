from dotenv import load_dotenv
from fire import Fire
from pprint import pprint
from functools import partial
from typing import *
import os

load_dotenv(".env.secret")

import pandas as pd

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, evaluate_existing
from langsmith import Client
from langsmith.schemas import Example, Run


# Constants
MODE = "evaluate"
MODEL_NAME = "gpt-3.5-turbo"
ACTION_LABELS = ["EXIT", "ASK_FOR_HINT", "MEASURE"]
TEMPERATURE = 1.5
NUM_GENERATIONS_PER_SAMPLE = 1
SWEEP_NAME = "persistsim-sweep-2"
NEW_EXPERIMENT_PREFIX = "trial-1"
EXISTING_EXPERIMENT_ID = "trial-1-68478e79"
SPLIT = "calibration"  # "evaluation"
PROMPT_NAME = "msamogh/persistsim-trial"

# Initialize client
client = Client()

# Load components
prompt = hub.pull(PROMPT_NAME)
chat_model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
output_parser = StrOutputParser()


# Define summary evaluator
def percentage_of_action(
    runs: list[Run], examples: list[Example], action_label: Text
) -> dict:
    action_count = 0
    for i, run in enumerate(runs):
        if run.outputs["output"] == action_label:
            action_count += 1
    return {
        "key": f"{action_label.lower()}_percentage",
        "score": action_count / len(runs),
    }


def predict_over_dataset(dataset: Text, dataset_filters: dict):
    # Run evaluation for the first time
    chain = prompt | chat_model | output_parser
    results = evaluate(
        chain.invoke,
        data=client.list_examples(dataset_name=dataset, metadata=dataset_filters),
        experiment_prefix=NEW_EXPERIMENT_PREFIX,
        num_repetitions=NUM_GENERATIONS_PER_SAMPLE,
    )
    return results


def next_action_distribution(action_labels: List[Text]):
    results = evaluate_existing(
        EXISTING_EXPERIMENT_ID,
        summary_evaluators=[
            partial(percentage_of_action, action_label=action_label)
            for action_label in action_labels
        ],
    )
    return {result.key: result.score for result in results._summary_results["results"]}


def main():
    if MODE == "predict":
        predict_over_dataset(SWEEP_NAME, {"split": SPLIT})
    elif MODE == "evaluate":
        distribution = {
            "experiment_id": EXISTING_EXPERIMENT_ID,
            **next_action_distribution(ACTION_LABELS),
        }
        # Read the list of distributions from a CSV onto a Pandas DataFrame, insert the new distribution, and write back to the CSV
        if os.path.exists("results/action_distribution.csv"):
            try:
                existing = pd.read_csv("results/action_distribution.csv")
                existing = pd.concat([existing, pd.DataFrame([distribution])])
                existing.to_csv("results/action_distribution.csv", index=False)
            except Exception as e:
                print(f"Exception: {e}")
                df = pd.DataFrame([distribution])
                df.to_csv("results/action_distribution.csv", index=False)
        else:
            df = pd.DataFrame([distribution])
            df.to_csv("results/action_distribution.csv", index=False)
    else:
        raise ValueError(f"Invalid mode: {MODE}")


if __name__ == "__main__":
    Fire(main)
