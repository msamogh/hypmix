from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt


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
    df["PREFIX"] = df["EXPERIMENT_ID"].str.split("-").str[:-1].str.join("-")

    # Add an extra "UNPREDICTED" column containing the remaining percentages
    df["UNPREDICTED"] = 1.0 - df.drop(columns=["EXPERIMENT_ID", "PREFIX"]).sum(axis=1)

    df = df.drop(columns=["EXPERIMENT_ID"])

    # Group by experiment_id and sum the percentages for each action
    grouped_df = df.groupby("PREFIX").mean()

    # Save the plot to figures/<experiment_id>.png
    for experiment_id in grouped_df.index:
        fig, ax = plt.subplots(figsize=(12, 8))
        grouped_df[grouped_df.index == experiment_id].T.plot(
            kind="bar", ax=ax, legend=False
        )
        # grouped_df[].T.plot(kind='bar', ax=ax, legend=False)
        ax.set_title(
            "Distribution of the Next Action in the Experiment Grouped by Experiment"
        )
        ax.set_xlabel("Action")
        ax.set_ylabel("Percentage")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()

        # Save the figure
        output_path = f"figures/{experiment_id}.png"
        fig.savefig(output_path)


if __name__ == "__main__":
    replot_figures()
