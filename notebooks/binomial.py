import numpy as np

# Parameters
n = 10      # Number of binary variables
k = 4       # Desired mean sum
sigma = 1   # Standard deviation for the normal distribution

# Sample the sum S from a normal distribution
S = int(np.random.normal(k, sigma))
S = max(0, min(S, n))  # Ensure S is within the valid range [0, n]

# Calculate the probability p
p = S / n

# Generate binary variables
X = np.random.binomial(1, p, n)

print(f"Sampled sum S: {S}")
print(f"Probability p: {p}")
print(f"Generated binary variables: {X}")
print(f"Sum of generated binary variables: {np.sum(X)}")
