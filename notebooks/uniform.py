import numpy as np
from scipy.stats import chisquare

# Example list of values (e.g., categorical data)
values = [1, 1, 1, 2, 2, 1, 2, 3, 1, 3, 3, 1, 3, 11, 1, 3, 1, 2, 1, 1]

# Count the frequency of each unique value
unique_values, observed_frequencies = np.unique(values, return_counts=True)

# Calculate the expected frequency if all frequencies were equal
expected_frequency = len(values) / len(unique_values)
expected_frequencies = np.full(len(unique_values), expected_frequency)

# Perform the Chi-Square Goodness of Fit test
chi2_statistic, p_value = chisquare(observed_frequencies, f_exp=expected_frequencies)

# Output the result
print("Chi-Square Statistic:", chi2_statistic)
print("P-value:", p_value)

# Interpret the result
alpha = 0.05  # significance level
if p_value < alpha:
    print("Reject the null hypothesis: The frequencies are not equal.")
else:
    print("Fail to reject the null hypothesis: The frequencies are equal.")
