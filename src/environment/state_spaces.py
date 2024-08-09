import random
from dataclasses import dataclass, field
from typing import *
from itertools import combinations

import numpy as np

random.seed(42)
np.random.seed(42)
rng = np.random.default_rng()


@dataclass
class State:
    state_space_name: str

    @property
    def state_variables(self) -> Dict[str, Any]:
        dict_vars = self.__dict__.copy()
        # Delete "state_space_name" key if it exists
        if "state_space_name" in dict_vars:
            dict_vars.pop("state_space_name")
        return dict_vars

    def describe_state(self) -> str:
        raise NotImplementedError


@dataclass
class StateSweep:
    state_space_name: str
    states: List[State]

    def describe_state(self) -> str:
        return "\n".join([state.describe_state() for state in self.states])


@dataclass
class HOStateA(State):

    state_space_name: str = "HOStateSpaceA"
    time_elapsed_in_minutes: int = 0
    num_submission_attempts: int = 0

    def describe_state(self) -> str:
        return f"""State:\nTIME_ELAPSED (the number of minutes that have passed since the start of the session): {self.time_elapsed_in_minutes} minutes\nNUM_SUBMISSION_ATTEMPTS (the number of times the user has submitted an answer since the start of the session): {self.num_submission_attempts}"""


@dataclass
class HOStateB(State):

    state_space_name: str = "HOStateSpaceB"
    num_submission_attempts: int = 0
    AP: int = 0
    AF1: int = 0
    AF2: int = 0
    AX: int = 0
    F1P: int = 0
    F1F2: int = 0
    F1X: int = 0
    F2P: int = 0
    F2X: int = 0
    PX: int = 0

    def describe_state(self) -> str:
        return f"""NUM_SUBMISSION_ATTEMPTS (the number of times the user has submitted an answer since the start of the session): {self.num_submission_attempts}
AP: {self.AP} (whether the user has measured the distance between Aphelion and Perihelion)
AF1: {self.AF1} (whether the user has measured the distance between Aphelion and Focus 1)
AF2: {self.AF2} (whether the user has measured the distance between Aphelion and Focus 2)
AX: {self.AX} (whether the user has measured the distance between Aphelion and the point X on the orbit)
F1P: {self.F1P} (whether the user has measured the distance between Focus 1 and Perihelion)
F1F2: {self.F1F2} (whether the user has measured the distance between Focus 1 and Focus 2)
F1X: {self.F1X} (whether the user has measured the distance between Focus 1 and the point X on the orbit)
F2P: {self.F2P} (whether the user has measured the distance between Focus 2 and Perihelion)
F2X: {self.F2X} (whether the user has measured the distance between Focus 2 and the point X on the orbit)
PX: {self.PX} (whether the user has measured the distance between Perihelion and the point X on the orbit)"""

    @staticmethod
    def generate_state_from_vector(state: np.ndarray) -> "HOStateB":
        state = state.tolist()
        return HOStateB(
            num_submission_attempts=state[0],
            AP=state[1],
            AF1=state[2],
            AF2=state[3],
            AX=state[4],
            F1P=state[5],
            F1F2=state[6],
            F1X=state[7],
            F2P=state[8],
            F2X=state[9],
            PX=state[10],
        )

    @staticmethod
    def generate_states(num_state_vars=10, std_dev=1) -> StateSweep:
        result_array = None
        # Sample the total number of measurements made so far (S) from a normal distribution
        for mean in range(0, 11):
            S = int(np.random.normal(mean, std_dev))
            S = max(
                0, min(S, num_state_vars)
            )  # Ensure S is within the valid range [0, n]

            # Calculate the probability p
            p = S / num_state_vars

            # Generate a random count vector that corresponds to number of measurements of *each type* so that it approximately sums up to S (binomial distribution with parameter S).
            if result_array is None:
                # Initialize the state space as a 2D array
                result_array = np.hstack(
                    (
                        np.tile([mean], (10, 1)),
                        np.random.binomial(1, p, (10, num_state_vars)),
                    )
                )
            else:
                # Add a new state as a new row
                result_array = np.vstack(
                    (
                        result_array,
                        np.hstack(
                            (
                                np.tile([mean], (10, 1)),
                                np.random.binomial(1, p, (10, num_state_vars)),
                            )
                        ),
                    )
                )
        # Generate state objects from row of the 2D result_array
        return StateSweep(
            state_space_name="binomial_prior",
            states=[
                HOStateB.generate_state_from_vector(state) for state in result_array
            ],
        )

    @staticmethod
    def generate_thorough_state_space(ks, vector_size=10) -> StateSweep:
        """
        Generates a 2D array where each row is a vector of size 'vector_size' with exactly 'k' ones,
        and an appended 11th element that's a random integer increasing with 'k'.

        Parameters:
        - ks (list[int]): Each k in ks is a number of ones in each vector.
        - vector_size (int): Size of the vector (default is 10).

        Returns:
        - np.ndarray: A 2D array of shape (C(vector_size, k), vector_size + 1).
        """
        k_results = []
        for k in ks:
            if not (0 <= k <= vector_size):
                raise ValueError(f"k must be between 0 and {vector_size} (inclusive).")

            # Step 1: Generate all combinations of positions for ones
            index_combinations = list(combinations(range(vector_size), k))
            num_combinations = len(index_combinations)

            # Initialize the array with zeros
            combinations_array = np.zeros((num_combinations, vector_size), dtype=int)

            # Set ones at the specified positions
            for idx, positions in enumerate(index_combinations):
                combinations_array[idx, list(positions)] = 1

            # Step 2: Define the increasing function for the appended element
            def increasing_function(sum_of_ones):
                if sum_of_ones < 2:
                    return 0  # Return a small random integer if sum_of_ones is 0
                else:
                    return sum_of_ones + np.random.randint(
                        0, int(0.3 * (sum_of_ones + 1)) + 1
                    )

            # Since sum_of_ones is always k, we can compute the appended values directly
            appended_values = np.array(
                [increasing_function(k) for _ in range(num_combinations)]
            ).reshape(-1, 1)

            # Concatenate the appended values to the combinations array
            result = np.hstack((appended_values, combinations_array))
            k_results.append(result)
        ks_result = np.vstack(k_results)
        return StateSweep(
            state_space_name="thorough_k",
            states=[HOStateB.generate_state_from_vector(row) for row in ks_result],
        )

    @staticmethod
    def generate_toy_state_space(start_idx=20, num_samples=10) -> StateSweep:
        state_sweep = HOStateB.generate_states()
        return StateSweep(
            state_space_name="toy_state_space",
            states=state_sweep.states[start_idx : start_idx + num_samples],
        )

    @staticmethod
    def generate_state_space_for_uniform_hyp() -> StateSweep:
        states = np.array(HOStateB.generate_states().states)
        return StateSweep(
            state_space_name="uniform_hyp_1",
            states=np.random.choice(states, int(0.8 * len(states)), replace=False),
        )

    @staticmethod
    def generate_uniform_state_space(size: str = "small") -> StateSweep:
        if size == "tiny":
            states = np.array(HOStateB.generate_states().states)[[0, 50, 100]]
        elif size == "small":
            states = np.array(HOStateB.generate_states().states)[
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            ]
        elif size == "medium":
            states = np.array(HOStateB.generate_states().states)[
                [
                    0,
                    5,
                    10,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                    45,
                    50,
                    55,
                    60,
                    65,
                    70,
                    75,
                    80,
                    85,
                    90,
                    95,
                    100,
                ]
            ]
        else:
            raise ValueError
        return StateSweep(state_space_name=f"uniform_state_space_{size}", states=states)
