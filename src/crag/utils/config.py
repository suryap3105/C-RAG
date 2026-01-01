
import yaml
from typing import Dict, Any
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Merges two configs. Override takes precedence.
    Deep merge could be implemented here if needed.
    """
    # Shallow merge for v1
    base.update(override)
    return base
