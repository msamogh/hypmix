from typing import *
from pprint import pprint
from dataclasses import dataclass

# Import OpenAI API
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(".env.secret")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def load_prompt(
    system_prompt: Text = "simple",
    learning_environment: Text = "simple",
    socratic_buddy: Text = "simple",
    learning_task: Text = "simple",
    action_space: Text = "simple",
    scenario: Text = "simple",
    final_instruction: Text = "simple",
) -> Dict[str, str]:
    # List of all variable names and values
    variables = {
        "system_prompt": system_prompt,
        "learning_environment": learning_environment,
        "socratic_buddy": socratic_buddy,
        "learning_task": learning_task,
        "action_space": action_space,
        "scenario": scenario,
        "final_instruction": final_instruction,
    }
    
    # Generate a map of each variable to its file contents
    result = {}
    for variable_name, variable_value in variables.items():
        try:
            with open(f"prompts/{variable_name}/{variable_value}.txt", 'r') as file:
                result[variable_name] = file.read()
        except FileNotFoundError:
            result[variable_name] = None

    # Concatenate them all into a single string separated by newlines
    return "\n".join(list(result.values()))

@dataclass
class Experiment:
    system_prompt: Text
    learning_environment: Text
    socratic_buddy: Text
    learning_task: Text
    action_space: Text
    scenario: Text
    final_instruction: Text

    def load_prompt(self):
        return load_prompt(
            system_prompt=self.system_prompt,
            learning_environment=self.learning_environment,
            socratic_buddy=self.socratic_buddy,
            learning_task=self.learning_task,
            action_space=self.action_space,
            scenario=self.scenario,
            final_instruction=self.final_instruction,
        )

    def generate(self):
        prompt = self.load_prompt()
        outputs = []
        scenarios = get_scenarios()
        learners = get_learners()
        for lc in learners:
            for scenario in scenarios:
                for key, value in scenario.items():
                    prompt = prompt.replace(f'{{{key}}}', str(value))
                for key, value in lc.items():
                    prompt = prompt.replace(f'{{{key}}}', str(value))
                output = gpt4(prompt).choices[0].message.content
                outputs.append({
                    "learner": lc,
                    "scenario": scenario,
                    "output": output,
                })
        return outputs

def get_learners() -> Dict[Text, Any]:
    return [
        {
            "persistence_description": "The ability to persist through challenges is low",
            "geometry_proficiency": "Geometry proficiency is low",
        },
        # {
        #     "persistence_description": "Persistence is high",
        #     "geometry_proficiency": "Geometry proficiency is low",
        # },
        # {
        #     "persistence_description": "Persistence is low",
        #     "geometry_proficiency": "Geometry proficiency is high",
        # },
        # {
        #     "persistence_description": "Persistence is high",
        #     "geometry_proficiency": "Geometry proficiency is high",
        # }
    ]


def get_scenarios() -> Dict[Text, Any]:
    return [
        # {
        #     "scenario": "Session just started",
        #     "time_elapsed": "1 minute",
        #     "num_submission_attempts": "2",
        #     "num_measurements_done_between_attempts": ["1", "2"],
        # },
        {
            "scenario": "Session has been going on for a long time now",
            "time_elapsed": "90 minutes",
            "num_submission_attempts": "10",
            "num_measurements_done_between_attempts": [],
        }
    ]


def gpt4(prompt: Text) -> Text:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        temperature=2
        )
    return response

if __name__ == "__main__":
    prompt = load_prompt(
        system_prompt="system_prompt",
        learning_environment="learning_environment",
        socratic_buddy="socratic_buddy",
        learning_task="learning_task",
        action_space="action_space",
        scenario="scenario",
        final_instruction="final_instruction",
    )
    outputs = []
    scenarios = get_scenarios()
    learners = get_learners()
    for lc in learners:
        for scenario in scenarios:
            for key, value in scenario.items():
                prompt = prompt.replace(f'{{{key}}}', str(value))
                for key, value in lc.items():
                    prompt = prompt.replace(f'{{{key}}}', str(value))
                output = gpt4(prompt).choices[0].message.content
                outputs.append({
                    "learner": lc,
                    "scenario": scenario,
                    "output": output,
                })
    # Convert JSON to CSV
    import csv
    with open("outputs.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)
    pprint(outputs)
