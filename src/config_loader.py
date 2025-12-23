import yaml

def load_config(file_path: str) -> dict:
    """
    Loads a yaml config file and returns a dictionary
    """
    with open(file_path, 'r') as file_obj:
        return yaml.safe_load(file_obj)