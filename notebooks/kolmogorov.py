import numpy as np
from scipy import stats

# Generate a sample of data
np.random.seed(42)  # For reproducibility
sample_data = [0.2, 0.4, 0.8, 0.5, 0.9]  # 100 data points uniformly distributed between 0 and 1

# Perform the Kolmogorov-Smirnov test
ks_statistic, p_value = stats.kstest(sample_data, "uniform")

# Print the results
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05  # Significance level
if p_value > alpha:
    print(
        "Fail to reject the null hypothesis: The sample follows a uniform distribution."
    )
else:
    print(
        "Reject the null hypothesis: The sample does not follow a uniform distribution."
    )
