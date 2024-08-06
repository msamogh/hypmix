import numpy as np
from scipy.stats import chisquare


def is_uniform_distribution(values, alpha=0.05):
    """
    Test if a set of values over n categories follows a uniform distribution.

    Parameters:
    values (list or np.ndarray): The observed frequencies in each category.
    alpha (float): The significance level (default is 0.05).

    Returns:
    bool: True if the distribution is uniform, False otherwise.
    """
    # Convert values to a numpy array if they are not already
    values = np.array(values)

    # The expected frequencies for a uniform distribution
    expected = np.full_like(values, np.mean(values))

    # Perform the chi-square goodness-of-fit test
    chi2_stat, p_value = chisquare(
        values, f_exp=np.sum(values) / np.sum(expected) * np.sum(expected)
    )

    # Return True if p_value is greater than alpha, indicating we fail to reject the null hypothesis
    return p_value > alpha


# Example usage
values = [10, 19, 8, 11, 9]
result = is_uniform_distribution(values)
print("Is uniform distribution:", result)
