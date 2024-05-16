from dataclasses import dataclass
from typing import Dict

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
    def create(cls, label: str, **variables) -> 'Scenario':
        variable_templates = {
            'time_elapsed': {
                'default': "Time elapsed: {value} minutes",
                'hours': "Hours passed: {value} hours"
            },
            'number_of_measurements': {
                'default': "Number of measurements: {value}",
                'count': "Measurement count: {value}"
            }
        }
        
        populated_vars = {}
        for var_name, (value, template_name) in variables.items():
            if var_name in variable_templates:
                state_var = StateVar(variable_templates[var_name])
                populated_vars[var_name] = state_var.populate(value, template_name)
            else:
                raise ValueError(f"No templates available for variable '{var_name}'")
        
        return cls(label=label, variables=populated_vars)

# Example usage with named arguments
scenarios = [
    Scenario.create(
        "scenario1", 
        time_elapsed=(10, 'default'), 
        number_of_measurements=(5, 'default')
    ),
    Scenario.create(
        "scenario2", 
        time_elapsed=(2, 'hours'), 
        number_of_measurements=(30, 'count')
    )
]
