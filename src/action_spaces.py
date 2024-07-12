from typing import *
from dataclasses import dataclass, field

from langsmith.schemas import Example, Run


@dataclass
class ActionSpace:

    action_space_name: str
    actions: Dict[Text, Text]

    def describe_action_space(self) -> str:
        return "\n".join(
            [
                f"{i+1}. {action}: {desc}"
                for i, (action, desc) in enumerate(self.actions.items())
            ]
        )


@dataclass
class HOActionSpaceA(ActionSpace):

    action_space_name: str = "HOActionSpaceA"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "SUBMIT": "The learner submits their answer.",
            "MEASURE": "Measure distance between two points.",
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
            "MEASURE-F1-Oi": "Measure distance between Focus 1 and the ith point on the orbit (i ranges from 1 to 10).",
            "MEASURE-A-F1": "Measures distance between Aphelion and Focus 1.",
            "MEASURE-A-P": "Measure distance between Aphelion and Perihelion.",
            "MEASURE-A-F2": "Measure distance between Aphelion and Focus 2.",
            "MEASURE-A-Oi": "Measure distance between Aphelion and the ith point on the orbit (i ranges from 1 to 10).",
            "MEASURE-F1-P": "Measure distance between Focus 1 and Perihelion.",
            "MEASURE-F1-F2": "Measure distance between Focus 1 and Focus 2.",
            "MEASURE-F2-P": "Measure distance between Focus 2 and Perihelion.",
            "MEASURE-F2-Oi": "Measure distance between Focus 2 and the ith point on the orbit (i ranges from 1 to 10).",
            "MEASURE-Oi-P": "Measure distance between the ith point on the orbit and Perihelion (i ranges from 1 to 10).",
            "SUBMIT(x, y, z)": "Submit a solution with three expressions x, y, and z as the answer. The expressions can be any arithmetic expression that involves the variables A-P, A-F1, A-F2, A-Oi, F1-P, F1-F2, F1-Oi, F2-P, F2-Oi, and Oi-P.",
            "EXIT": "Exit the session.",
            "UNPREDICTED": "Perform an action not in the action space.",
        }
    )

    @staticmethod
    def productive_measurement_percentage(
        runs: list[Run], examples: list[Example]
    ) -> bool:
        productive_actions = 0
        measure_actions = 0
        for i, run in enumerate(runs):
            if run.outputs is None:
                continue
            action_label = run.outputs["output"]
            if "MEASURE" in run.outputs["output"]:
                measure_actions += 1
                if action_label in [
                    "MEASURE-F1-Oi",
                    "MEASURE-F2-Oi",
                    "MEASURE-A-F1",
                    "MEASURE-A-F2",
                    "MEASURE-F1-P",
                    "MEASURE-F2-P",
                ]:
                    productive_actions += 1
        return {
            "key": "productive_actions_ratio",
            "score": productive_actions / measure_actions if measure_actions > 0 else 0,
        }

    @staticmethod
    def unproductive_measurement_percentage(
        runs: list[Run], examples: list[Example]
    ) -> bool:
        unproductive_actions = 0
        measure_actions = 0
        for i, run in enumerate(runs):
            if run.outputs is None:
                continue
            action_label = run.outputs["output"]
            if "MEASURE" in run.outputs["output"]:
                measure_actions += 1
                if action_label not in [
                    "MEASURE-F1-Oi",
                    "MEASURE-F2-Oi",
                    "MEASURE-A-F1",
                    "MEASURE-A-F2",
                    "MEASURE-F1-P",
                    "MEASURE-F2-P",
                ]:
                    unproductive_actions += 1
        return {
            "key": "unproductive_actions_ratio",
            "score": unproductive_actions / measure_actions if measure_actions > 0 else 0,
        }
