import json
from pathlib import Path

def read_config(file_path: str = "pipeline.json") -> dict:
    """
    Read the pipeline configuration JSON file with error handling.
    
    Args:
        file_path: Path to the JSON configuration file
        
    Returns:
        dict: Configuration dictionary, or empty dict if error
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' not found")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return {}