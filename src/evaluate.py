from dotenv import load_dotenv
from fire import Fire
from pprint import pprint
from functools import partial
from typing import *

load_dotenv(".env.secret")

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, evaluate_existing
from langsmith import Client
from langsmith.schemas import Example, Run


# Constants
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 1.5
NUM_GENERATIONS_PER_SAMPLE = 1
SWEEP_NAME = "persistsim-sweep-2"
NEW_EXPERIMENT_PREFIX = "trial-1"
EXISTING_EXPERIMENT_ID = "trial-1-976b0bc6"
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
    exit_count = 0
    for i, run in enumerate(runs):
        if run.outputs["output"] == action_label:
            exit_count += 1
    return {
        "key": f"{action_label.lower()}_percentage",
        "score": exit_count / len(runs),
    }


def predict(split: Text):
    # Run evaluation for the first time
    chain = prompt | chat_model | output_parser
    results = evaluate(
        chain.invoke,
        data=client.list_examples(dataset_name=SWEEP_NAME, metadata={"split": SPLIT}),
        experiment_prefix=NEW_EXPERIMENT_PREFIX,
        num_repetitions=NUM_GENERATIONS_PER_SAMPLE,
    )
    pprint(results)


def evaluate(split: Text, action_label: Text):
    results = evaluate_existing(
        EXISTING_EXPERIMENT_ID,
        summary_evaluators=[partial(percentage_of_action, action_label=action_label)],
    )
    pprint(results)


def main(
    mode: str = "predict",
    split: Text = "calibration",
    action_label: Text = "ASK_FOR_HINT",
):
    if mode == "predict":
        predict(split)
    elif mode == "evaluate":
        evaluate(split, action_label)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    Fire(main)
