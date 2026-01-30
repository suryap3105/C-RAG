"""
Configuration Validation and Management
"""
import yaml
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Configuration validation rule."""
    path: str  # Dot-separated path (e.g., "gnn.hidden_channels")
    required: bool = True
    type_check: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None


class ConfigValidator:
    """
    Validate configuration files.
    """
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default validation rules."""
        # Graph config
        self.add_rule("graph.nodes_path", type_check=str)
        self.add_rule("graph.edges_path", type_check=str)
        
        # Partitioning
        self.add_rule("partitioning.enabled", required=True, type_check=bool)
        self.add_rule("partitioning.method", allowed_values=['metis', 'leiden', 'spectral'])
        self.add_rule("partitioning.n_partitions", type_check=int, min_value=1, max_value=1000)
        
        # Vector store
        self.add_rule("vector_store.embedding_dim", type_check=int, allowed_values=[768, 384, 512, 1024])
        self.add_rule("vector_store.index_type", allowed_values=['Flat', 'IVF', 'HNSW'])
        
        # GNN
        self.add_rule("gnn.in_channels", type_check=int, min_value=64, max_value=2048)
        self.add_rule("gnn.hidden_channels", type_check=int, min_value=32, max_value=1024)
        self.add_rule("gnn.out_channels", type_check=int, min_value=32, max_value=1024)
        self.add_rule("gnn.num_gat_layers", type_check=int, min_value=1, max_value=10)
        self.add_rule("gnn.num_gin_layers", type_check=int, min_value=1, max_value=10)
        self.add_rule("gnn.dropout", type_check=float, min_value=0.0, max_value=0.9)
        
        # LLM
        self.add_rule("llm.provider", allowed_values=['mock', 'ollama', 'openai', 'anthropic'])
        self.add_rule("llm.temperature", type_check=float, min_value=0.0, max_value=2.0)
        self.add_rule("llm.max_tokens", type_check=int, min_value=1, max_value=100000)
        
        # Training
        self.add_rule("training.epochs", type_check=int, min_value=1, max_value=10000)
        self.add_rule("training.batch_size", type_check=int, min_value=1, max_value=1024)
        self.add_rule("training.learning_rate", type_check=float, min_value=1e-6, max_value=1.0)
        
    def add_rule(self, path: str, **kwargs):
        """Add a validation rule."""
        self.rules.append(ValidationRule(path=path, **kwargs))
        
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        for rule in self.rules:
            value = self._get_nested_value(config, rule.path)
            
            # Check required
            if value is None and rule.required:
                errors.append(f"Missing required field: {rule.path}")
                continue
                
            if value is None:
                continue
                
            # Type check
            if rule.type_check and not isinstance(value, rule.type_check):
                errors.append(
                    f"{rule.path}: Expected {rule.type_check.__name__}, "
                    f"got {type(value).__name__}"
                )
                
            # Range check
            if rule.min_value is not None and value < rule.min_value:
                errors.append(f"{rule.path}: Value {value} below minimum {rule.min_value}")
                
            if rule.max_value is not None and value > rule.max_value:
                errors.append(f"{rule.path}: Value {value} above maximum {rule.max_value}")
                
            # Allowed values
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(
                    f"{rule.path}: Value '{value}' not in allowed values: {rule.allowed_values}"
                )
                
            # Custom validator
            if rule.custom_validator:
                try:
                    rule.custom_validator(value)
                except Exception as e:
                    errors.append(f"{rule.path}: Custom validation failed: {e}")
                    
        return errors
        
    @staticmethod
    def _get_nested_value(d: Dict, path: str) -> Any:
        """Get nested dictionary value by dot-separated path."""
        parts = path.split('.')
        current = d
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
            
        return current
        
    def validate_file(self, path: str) -> List[str]:
        """Validate configuration file."""
        config = self._load_config(path)
        return self.validate(config)
        
    @staticmethod
    def _load_config(path: str) -> Dict:
        """Load configuration file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)


class ConfigMerger:
    """
    Merge multiple configuration sources.
    """
    @staticmethod
    def merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger.merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    @staticmethod
    def load_with_overrides(
        base_path: str,
        override_paths: List[str] = None,
        env_overrides: Dict[str, Any] = None
    ) -> Dict:
        """
        Load configuration with multiple override layers.
        
        Args:
            base_path: Base configuration file
            override_paths: Additional config files to merge
            env_overrides: Environment-specific overrides
            
        Returns:
            Merged configuration
        """
        # Load base
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Apply file overrides
        if override_paths:
            for path in override_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        override = yaml.safe_load(f)
                        config = ConfigMerger.merge(config, override)
                        
        # Apply environment overrides
        if env_overrides:
            config = ConfigMerger.merge(config, env_overrides)
            
        return config


class ConfigTemplate:
    """
    Generate configuration templates.
    """
    @staticmethod
    def generate_template(output_path: str, include_comments: bool = True):
        """Generate a configuration template."""
        template = {
            "graph": {
                "nodes_path": "data/nodes.jsonl",
                "edges_path": "data/edges.jsonl"
            },
            "partitioning": {
                "enabled": True,
                "method": "metis",
                "n_partitions": 10
            },
            "vector_store": {
                "embedding_dim": 768,
                "index_type": "Flat"
            },
            "gnn": {
                "in_channels": 768,
                "hidden_channels": 256,
                "out_channels": 256,
                "num_gat_layers": 2,
                "num_gin_layers": 4,
                "dropout": 0.1
            },
            "llm": {
                "provider": "mock",
                "model": "llama3.2",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "weight_decay": 0.00001
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Generated configuration template: {output_path}")


def validate_and_load(config_path: str) -> Dict[str, Any]:
    """
    Validate and load configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    validator = ConfigValidator()
    errors = validator.validate_file(config_path)
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
