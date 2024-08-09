import numpy as np
from itertools import combinations


def generate_combinations_and_append():
    # Step 1: Generate all possible combinations of 0 and 1 for a vector of size 10
    num_elements = 10
    combinations = np.array(np.meshgrid(*[[0, 1]] * num_elements)).T.reshape(
        -1, num_elements
    )

    # Step 2: Define a function to generate an increasing integer based on the sum of 1s
    def increasing_function(sum_of_ones):
        if sum_of_ones < 2:
            return 0  # Return a small random integer if sum_of_ones is 0
        else:
            return sum_of_ones + np.random.randint(0, int(0.3 * (sum_of_ones + 1)) + 1)

    # Step 3: Apply the function to each row and hstack it to the original array
    appended_column = np.array(
        [increasing_function(row.sum()) for row in combinations]
    ).reshape(-1, 1)
    result = np.hstack((combinations, appended_column))

    return result


# Example usage
resulting_array = generate_combinations_and_append()
breakpoint()
print(resulting_array)
