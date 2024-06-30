from typing import *

from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env.secret")
load_dotenv(".env")


def learner_characteristic_to_action_label_prob_map(
    dataset_name: Text,
    experiment_names: List[Text],
    learner_characteristic: Text,
    action_label: Text,
) -> Dict[Text, float]:
    mapping = dict()
    for dataset_name in experiment_names:
        df = pd.read_csv(f"results/{dataset_name}.csv")
        mapping[dataset_name] = df[learner_characteristic].values
    return mapping


if __name__ == "__main__":
    result = learner_characteristic_to_action_label_prob_map(
        "persistsim-sweep-14", ["experiment--de2bd8a6"], "geometry_proficiency_level"
    )
    print(result)
