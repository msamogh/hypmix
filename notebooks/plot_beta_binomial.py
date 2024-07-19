import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
import fire


def plot_beta_binomial(alpha=1, beta_param=3, n=10, num_samples=1000):
    """
    Plots the Beta-Binomial distribution given the parameters of the Beta distribution.

    Args:
        alpha (float): Alpha parameter of the Beta distribution.
        beta_param (float): Beta parameter of the Beta distribution.
        n (int): Number of binary variables.
        num_samples (int): Number of samples to generate for the plot.
    """
    # Generate num_samples of Y from Beta distribution
    Y_samples = beta.rvs(alpha, beta_param, size=num_samples)

    # Generate binary variables from Binomial distribution for each Y sample
    X_samples = [binom.rvs(1, y, size=n) for y in Y_samples]

    # Count occurrences of each sum of binary variables
    sum_X_samples = [np.sum(x) for x in X_samples]
    counts = np.bincount(sum_X_samples, minlength=n + 1)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(n + 1), counts / num_samples, color="skyblue")
    plt.xlabel("Number of successes (sum of binary variables)")
    plt.ylabel("Probability")
    plt.title(f"Beta-Binomial Distribution (alpha={alpha}, beta={beta_param}, n={n})")
    plt.xticks(range(n + 1))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    fire.Fire(plot_beta_binomial)
