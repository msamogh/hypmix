from typing import *

from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env.secret")
load_dotenv(".env")


def learner_characteristic_to_action_label_prob_map(
    dataset_name: Text,
    learner_characteristic: Text,
    row_fn: Callable[[pd.Series], Any],
) -> Dict[Text, float]:
    """Returns a dict of <learner_characteristic> to <cumulative_action_label_prob> for a given learner_characteristic."""
    df = pd.read_csv(f"results/{dataset_name}.csv")
    df = df.groupby(learner_characteristic).apply(row_fn, include_groups=False)
    return {df.iloc[0].name: df.iloc[0][0]}


if __name__ == "__main__":

    def percentage_productive_measurements(row):
        return (
            row["action_measure-f1-oi_percentage"]
            + row["action_measure-f2-oi_percentage"]
            + row["action_measure-a-f1_percentage"]
            + row["action_measure-a-f2_percentage"]
            + row["action_measure-f1-p_percentage"]
            + row["action_measure-f2-p_percentage"]
        )

    def percentage_unproductive_measurements(row):
        return (
            row["action_measure-a-p_percentage"]
            + row["action_measure-a-oi_percentage"]
            + row["action_measure-f1-f2_percentage"]
            + row["action_measure-oi-p_percentage"]
        )

    DATASET_NAME = "persistsim-sweep-18"

    result = learner_characteristic_to_action_label_prob_map(
        DATASET_NAME,
        "geometry_proficiency_levels",
        percentage_productive_measurements,
    )
    print(f"Percentage productive measurements: {result}")
    result = learner_characteristic_to_action_label_prob_map(
        DATASET_NAME,
        "geometry_proficiency_levels",
        percentage_unproductive_measurements,
    )
    print(f"Percentage unproductive measurements: {result}")
