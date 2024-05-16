import yaml

class YamlData:
    def __init__(self, file_path):
        self.data = self.load_yaml(file_path)
        
    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def get_value(self, *indices):
        """
        Fetch values from nested YAML data using a list of indices.
        Returns the string directly if the path ends at a dictionary key with a string value.
        """
        result = self.data['definitions']  # Starting point is always the definitions list
        key_path = ['generic_behaviors', 'holo_behaviors']  # Path to traverse inside nested structures
        
        try:
            for level, index in enumerate(indices):
                if isinstance(result, list):
                    result = result[index]
                elif isinstance(result, dict):
                    if 'definition' in result and (len(indices) == level + 1 or not key_path):
                        return result['definition']  # Return just the string for 'definition'
                    elif key_path:
                        result = result[key_path[0]]
                        key_path.pop(0)
                        result = result[index]

            # If ending on a dictionary and no further keys to traverse, return the first value if it's a string
            if isinstance(result, dict):
                for key in result:
                    if isinstance(result[key], str):
                        return result[key]
            return result
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error navigating through indices {' -> '.join(map(str, indices))}: {e}")
            return None

# Example Usage:
if __name__ == "__main__":
    yaml_data = YamlData("optrees.yaml")
    print(yaml_data.get_value(1))  # Accessing just the definition string of the first definition
    print(yaml_data.get_value(1, 0, 0))  # Accessing the first holo_behavior of the first generic_behavior of the second definition
