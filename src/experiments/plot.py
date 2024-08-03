from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd


def replot_figures():

    # Sample CSV data provided by the user
    csv_data = open("results/action_distribution.csv").read()

    # Read the sample data into a DataFrame
    df = pd.read_csv(StringIO(csv_data))

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Rename the columns to remove "_percentage" and make them uppercase
    df.columns = [col.replace("_percentage", "").upper() for col in df.columns]

    # Add new column called 'prefix' that contains the initial substring of the experiment_id until the last hyphen
    df["PREFIX"] = df["EXPERIMENT_ID"].str.split("-").str[-1].str.join("-")

    # Add an extra "UNPREDICTED" column containing the remaining percentages
    # Subtract only those columns starting with "ACTION_"
    df["UNPREDICTED"] = 1.0 - df.filter(like="ACTION_").drop(
        columns=["ACTION_SPACE"]
    ).sum(axis=1)

    # Drop "ACTION_SPACE"
    df = df.drop(columns=["ACTION_SPACE"])

    # Only retain "PREFIX" and columns starting with "ACTION_"
    df = df[
        ["PREFIX", "SPLIT"]
        + df.filter(like="ACTION_").columns.tolist()
        + ["UNPREDICTED"]
    ]

    # Remove '-'s from "PREFIX"
    df["PREFIX"] = df["PREFIX"].str.replace("-", "")

    # Group by experiment_id and sum the percentages for each action
    grouped_df = df.groupby(["PREFIX", "SPLIT"]).mean()

    # Save the plot to figures/<experiment_id>.png
    for index in grouped_df.index:
        experiment_id, split = index
        fig, ax = plt.subplots(figsize=(12, 8))
        grouped_df[grouped_df.index == index].T.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(
            "Distribution of the Next Action in the Experiment Grouped by Experiment"
        )
        ax.set_xlabel("Action")
        ax.set_ylabel("Percentage")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()

        # Save the figure
        output_path = f"figures/{experiment_id}_{split}.png"
        fig.savefig(output_path)


if __name__ == "__main__":
    replot_figures()
