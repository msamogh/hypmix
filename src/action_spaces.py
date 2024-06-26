from typing import *
from dataclasses import dataclass, field


@dataclass
class ActionSpace:

    action_space_name: str
    actions: Dict[Text, Text]

    def describe_action_space(self) -> str:
        return "\n".join([f"{action}: {desc}" for action, desc in self.actions.items()])


@dataclass
class HOActionSpaceA(ActionSpace):

    action_space_name: str = "HOActionSpaceA"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "SUBMIT": "The learner submits their answer.",
            "MEASURE": "The learner measures the distance between two points.",
            "USE_CALCULATOR": "The learner uses the calculator.",
            "ASK_FOR_HELP": "The learner asks for help.",
            "ASK_FOR_HINT": "The learner asks for a hint.",
            "ASK_FOR_EXPLANATION": "The learner asks for an explanation.",
            "ASK_FOR_SOLUTION": "The learner asks for a solution.",
            "EXIT": "The learner exits the session.",
        }
    )


@dataclass
class HOActionSpaceB(ActionSpace):

    action_space_name: str = "HOActionSpaceB"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "MEASURE-XY": "The learner measures the distance between two points X and Y. XY can be any pair of points from the set {AP, AF1, AF2, AOi, F1P, F1F2, F1Oi, F2P, F2O, OiP}. They stand for A - Aphelion, P - Perihelion, F1 - Focus 1, F2 - Focus 2, Oi - ith point on the orbit (i ranges from 1 to 10).",
            "ZOOM-IN": "The learner zooms in on the planetary system.",
            "ZOOM-OUT": "The learner zooms out on the planetary system.",
            "ROTATE": "The learner rotates the planetary system.",
            "PAN": "The learner pans the planetary system.",
            "SUBMIT": "The learner submits their solution.",
            "EXIT": "The learner exits the session.",
            "UNPREDICTED": "The learner performs an action that is not predicted by the model.",
        }
    )
