from dataclasses import dataclass, field
from typing import *

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
class HOActionSpace(ActionSpace):

    @property
    def productive_action_labels(self) -> list[str]:
        raise NotImplementedError

    @property
    def exit_action_label(self) -> str:
        raise NotImplementedError

    def is_measure_action(self, llm_prediction: str):
        raise NotImplementedError

    @property
    def productive_measurement_percentage(
        self, runs: list[Run], examples: list[Example]
    ) -> bool:
        productive_actions = 0
        measure_actions = 0
        for i, run in enumerate(runs):
            if run.outputs is None:
                continue
            action_label = run.outputs["output"]
            if self.is_measure_action(run.outputs["output"]):
                measure_actions += 1
                if (
                    action_label in self.productive_action_labels
                    and run.metadata.get(action_label, 0) != 1
                ):
                    productive_actions += 1
        return {
            "key": "productive_actions_ratio",
            "score": productive_actions / measure_actions if measure_actions > 0 else 0,
        }

    @property
    def unproductive_measurement_percentage(
        self, runs: list[Run], examples: list[Example]
    ) -> bool:
        unproductive_actions = 0
        measure_actions = 0
        for i, run in enumerate(runs):
            if run.outputs is None:
                continue
            action_label = run.outputs["output"]
            if self.is_measure_action(run.outputs["output"]):
                measure_actions += 1
                if (
                    action_label not in self.productive_action_labels
                    or run.metadata.get(action_label, 0) == 1
                ):
                    unproductive_actions += 1
        return {
            "key": "unproductive_actions_ratio",
            "score": (
                unproductive_actions / measure_actions if measure_actions > 0 else 0
            ),
        }


@dataclass
class HOActionSpaceB(ActionSpace):

    action_space_name: str = "HOActionSpaceB"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "MEASURE-F1-X": "Measure distance between Focus 1 and a specific point X on the orbit.",
            "MEASURE-A-F1": "Measures distance between Aphelion and Focus 1.",
            "MEASURE-A-P": "Measure distance between Aphelion and Perihelion.",
            "MEASURE-A-F2": "Measure distance between Aphelion and Focus 2.",
            "MEASURE-A-X": "Measure distance between Aphelion and a specific point X on the orbit.",
            "MEASURE-F1-P": "Measure distance between Focus 1 and Perihelion.",
            "MEASURE-F1-F2": "Measure distance between Focus 1 and Focus 2.",
            "MEASURE-F2-P": "Measure distance between Focus 2 and Perihelion.",
            "MEASURE-F2-X": "Measure distance between Focus 2 and a specific point X on the orbit.",
            "MEASURE-P-X": "Measure distance between Perihelion and a specific point X on the orbit.",
            "SUBMIT(e1, e2, e3)": "Submit a solution with three expressions e1, e2, and e3 as the answer. Each expression can be any arithmetic expression that involves the variables A-P, A-F1, A-F2, A-X, F1-P, F1-F2, F1-X, F2-P, F2-X, and P-X.",
            "EXIT": "Exit the session.",
            "UNPREDICTED": "Perform an action not in the action space.",
        }
    )

    @property
    def exit_action_label(self) -> str:
        return "EXIT"

    @property
    def productive_action_labels(self) -> list[str]:
        return [
            "MEASURE-F1-X",
            "MEASURE-F2-X",
            "MEASURE-A-F1",
            "MEASURE-A-F2",
            "MEASURE-F1-P",
            "MEASURE-F2-P",
        ]

    def is_measure_action(self, llm_prediction):
        return "MEASURE" in llm_prediction


@dataclass
class HOActionSpaceC(HOActionSpace):

    action_space_name: str = "HOActionSpaceC"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "MEASURE-X-F1": "Measure distance between a specific point X on the orbit and Focus 1.",
            "MEASURE-F1-A": "Measure distance between Focus 1 and Aphelion.",
            "MEASURE-P-A": "Measure distance between Perihelion and Aphelion.",
            "MEASURE-F2-A": "Measure distance between Focus 2 and Aphelion.",
            "MEASURE-X-A": "Measure distance between a specific point X on the orbit and Aphelion.",
            "MEASURE-P-F1": "Measure distance between Perihelion and Focus 1.",
            "MEASURE-F2-F1": "Measure distance between Focus 2 and Focus 1.",
            "MEASURE-P-F2": "Measure distance between Perihelion and Focus 2.",
            "MEASURE-X-F2": "Measure distance between a specific point X on the orbit and Focus 2.",
            "MEASURE-X-P": "Measure distance between a specific point X on the orbit and Perihelion.",
            "SUBMIT(e1, e2, e3)": "Submit a solution with three expressions e1, e2, and e3 as the answer. Each expression can be any arithmetic expression that involves the variables P-A, F1-A, F2-A, X-A, P-F1, F2-F1, X-F1, P-F2, X-F2, and X-P.",
            "QUIT": "Quit the session.",
            "UNPREDICTED": "Perform an action not in the action space.",
        }
    )

    @property
    def exit_action_label(self) -> str:
        return "QUIT"

    @property
    def productive_action_labels(self):
        return [
            "MEASURE-X-F1",
            "MEASURE-X-F2",
            "MEASURE-F1-A",
            "MEASURE-F2-A",
            "MEASURE-P-F1",
            "MEASURE-P-F2",
        ]

    def is_measure_action(self, llm_prediction):
        return "MEASURE" in llm_prediction


@dataclass
class HOActionSpaceD(HOActionSpace):

    action_space_name: str = "HOActionSpaceD"
    actions: Dict[Text, Text] = field(
        default_factory=lambda: {
            "CALC(f1, o)": "Calculate distance between Focus 1 and a random point on the orbit.",
            "CALC(a, f1)": "Calculate distance between Aphelion and Focus 1.",
            "CALC(a, p)": "Calculate distance between Aphelion and Perihelion.",
            "CALC(a, f2)": "Calculate distance between Aphelion and Focus 2.",
            "CALC(a, o)": "Calculate distance between Aphelion and a random point on the orbit.",
            "CALC(f1, p)": "Calculate distance between Focus 1 and Perihelion.",
            "CALC(f1, f2)": "Calculate distance between Focus 1 and Focus 2.",
            "CALC(f2, p)": "Calculate distance between Focus 2 and Perihelion.",
            "CALC(f2, x)": "Calculate distance between Focus 2 and a random point on the orbit.",
            "CALC(p, x)": "Calculate distance between Perhelion and a random point on the orbit and Perihelion.",
            "SUBMIT-SOLN(e1, e2, e3)": "Submit a solution with three expressions e1, e2, and e3 as the answer. The expressions can be any arithmetic expression that involves the distances, f1-o, a-f2, a-o, f1-p, f1-f2, f1-o, f2-p, f2-o, and o-p.",
            "QUIT": "Quit the learning session.",
            "UNPREDICTED": "Perform an action not in the action space.",
        }
    )

    @property
    def exit_action_label(self) -> str:
        return "QUIT"

    @property
    def productive_action_labels(self):
        return [
            "CALC(f1, o)",
            "CALC(f2, o)",
            "CALC(a, f1)",
            "CALC(a, f2)",
            "CALC(f1, p)",
            "CALC(f2, p)",
        ]

    def is_measure_action(self, llm_prediction):
        return "CALC" in llm_prediction
