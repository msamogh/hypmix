from dataclasses import dataclass
from typing import *


@dataclass
class LearningEnvironment:
    pass


@dataclass
class State:
    pass


@dataclass
class DecisionSample:
    state: State
    action: Text


@dataclass
class HOState(LearningEnvironment):
    pass


@dataclass
class HOStateA(HOState):
    num_minutes_elapsed: int
    # If the number of unique measurements is less than the number of measurements, 
    # the agent has repeated measurements, which could be an indicator of
    # the learner wheel-spinning.
    num_total_measurements: int
    num_unique_measurements: int
    num_submission_attempts: int
    num_time_spent_between_submissions: List[int]

    def to_string(self):
        # Newline-separated string representation of the state
        assert len(self.num_time_spent_between_submissions) == self.num_submission_attempts
        return f"Minutes Elapsed: {self.num_minutes_elapsed}\n" \
               f"Total Measurements: {self.num_total_measurements}\n" \
               f"Unique Measurements: {self.num_unique_measurements}\n" \
               f"Submission Attempts: {self.num_submission_attempts}\n" \
               f"Time Spent Between Submissions: {self.num_time_spent_between_submissions}"


@dataclass
class HODecisionSample(DecisionSample):
    state: State


@dataclass
class Dataset:
    samples: Sequence[HODecisionSample]
