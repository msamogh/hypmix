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
            "SUBMIT": "The user submits their answer.",
            "MEASURE": "The user measures the distance between two points.",
            "USE_CALCULATOR": "The user uses the calculator.",
            "ASK_FOR_HELP": "The user asks for help.",
            "ASK_FOR_HINT": "The user asks for a hint.",
            "ASK_FOR_EXPLANATION": "The user asks for an explanation.",
            "ASK_FOR_SOLUTION": "The user asks for a solution.",
            "EXIT": "The user exits the session.",
        }
    )
