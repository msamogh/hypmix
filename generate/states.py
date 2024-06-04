from dataclasses import dataclass
from typing import Dict, Text

@dataclass
class StateVar:
    templates: Dict[str, str]

    def populate(self, value: int, template_name: str = 'default') -> str:
        """Populate the specified template with a value, using a named key for the template."""
        try:
            return self.templates[template_name].format(value=value)
        except KeyError:
            raise ValueError(f"Template with name '{template_name}' not found")

@dataclass
class Scenario:
    label: str
    variables: Dict[str, str]

    @classmethod
    def create(cls, label: str, **variable_data) -> 'Scenario':
        variables = {}
        for key, (value, template_name) in variable_data.items():
            # Default templates for demonstration; these would typically be more comprehensive or externally defined
            templates = {
                'time_elapsed': {
                    'default': "Time elapsed: {value} minutes",
                    'hours': "Hours passed: {value} hours"
                },
                'number_of_measurements': {
                    'default': "Number of measurements: {value}",
                    'count': "Measurement count: {value}"
                }
            }
            if key in templates:
                state_var = StateVar(templates[key])
                variables[key] = state_var.populate(value, template_name)
            else:
                raise ValueError(f"No templates available for variable '{key}'")
        return cls(label=label, variables=variables)

# Example usage:
if __name__ == "__main__":
    scenario_data = {
        'time_elapsed': (90, 'hours'),
        'number_of_measurements': (15, 'count')
    }
    scenario = Scenario.create("Experiment 101", **scenario_data)
    print(scenario)
