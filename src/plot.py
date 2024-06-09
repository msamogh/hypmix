from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt


def replot_figures():

    # Sample CSV data provided by the user
    csv_data = open('results/action_distribution.csv').read()

    # Read the sample data into a DataFrame
    df = pd.read_csv(StringIO(csv_data))

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Rename the columns to remove "_percentage" and make them uppercase
    df.columns = [col.replace('_percentage', '').upper() for col in df.columns]

    # Add an extra "UNPREDICTED" column containing the remaining percentages
    df['UNPREDICTED'] = 1.0 - df.drop(columns=['EXPERIMENT_ID']).sum(axis=1)

    # Group by experiment_id and sum the percentages for each action
    grouped_df = df.groupby('EXPERIMENT_ID').sum()

    # Save the plot to figures/<experiment_id>.png
    experiment_id = grouped_df.index[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped_df.T.plot(kind='bar', ax=ax, legend=False)
    ax.set_title('Distribution of the Next Action in the Experiment Grouped by EXPERIMENT_ID')
    ax.set_xlabel('Action')
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    # Save the figure
    output_path = f"figures/{experiment_id}.png"
    fig.savefig(output_path)