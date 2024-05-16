from typing import *

# Import OpenAI API
import openai


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


def populate_scenario(state: Dict[Text, Text]) -> Text:
    pass





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
