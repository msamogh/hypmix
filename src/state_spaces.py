from typing import *
from dataclasses import dataclass, field


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
    AO: int = 0
    F1P: int = 0
    F1F2: int = 0
    F1O: int = 0
    F2P: int = 0
    F2O: int = 0
    OP: int = 0

    def describe_state(self) -> str:
        return f"""NUM_SUBMISSION_ATTEMPTS (the number of times the user has submitted an answer since the start of the session): {self.num_submission_attempts}
AP: {self.AP} (whether the user has measured the distance between Aphelion and Perihelion)
AF1: {self.AF1} (whether the user has measured the distance between Aphelion and Focus 1)
AF2: {self.AF2} (whether the user has measured the distance between Aphelion and Focus 2)
AO: {self.AO} (whether the user has measured the distance between Aphelion and the ith point on the orbit)
F1P: {self.F1P} (whether the user has measured the distance between Focus 1 and Perihelion)
F1F2: {self.F1F2} (whether the user has measured the distance between Focus 1 and Focus 2)
F1O: {self.F1O} (whether the user has measured the distance between Focus 1 and the ith point on the orbit)
F2P: {self.F2P} (whether the user has measured the distance between Focus 2 and Perihelion)
F2O: {self.F2O} (whether the user has measured the distance between Focus 2 and the ith point on the orbit)
OP: {self.OP} (whether the user has measured the distance between the ith point on the orbit and Perihelion)"""
    
    @staticmethod
    def NO_MEASUREMENTS() -> "HOStateB":
        return HOStateB.generate_state(0, {})

    @staticmethod
    def generate_state(
        num_submission_attempts, action_mask: Dict[str, bool]
    ) -> "HOStateB":
        return HOStateB(
            num_submission_attempts=num_submission_attempts,
            AP=action_mask.get("AP", 0),
            AF1=action_mask.get("AF1", 0),
            AF2=action_mask.get("AF2", 0),
            AO=action_mask.get("AO", 0),
            F1P=action_mask.get("F1P", 0),
            F1F2=action_mask.get("F1F2", 0),
            F1O=action_mask.get("F1O", 0),
            F2P=action_mask.get("F2P", 0),
            F2O=action_mask.get("F2O", 0),
            OP=action_mask.get("OP", 0),
        )
