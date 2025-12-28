"""
Model configuration utility for Neophysics.
Includes configuration for T5 model and Physics Engine.
"""

import os
import json
import psutil
from typing import Dict, Any


class ModelConfig:
    """Configuration manager for the physics engine."""
    
    def __init__(self):
        self.config_path = "models/model_config.json"
        self.config = self.load_config()
    
    def detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system RAM and CUDA availability."""
        capabilities = {
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 1),
            "has_cuda": False
        }
        
        try:
            import torch
            capabilities["has_cuda"] = torch.cuda.is_available()
            if capabilities["has_cuda"]:
                capabilities["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        except ImportError:
            pass
            
        return capabilities
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default T5 configuration."""
        return {
            "model_name": "t5-small",
            "batch_size": 4,
            "max_length": 256
        }
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.config = config
    
    def load_config(self) -> Dict[str, Any]:
        """Load saved configuration or return default."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()


def main():
    """Print system capabilities."""
    config = ModelConfig()
    caps = config.detect_system_capabilities()
    print("System Capabilities:")
    print(json.dumps(caps, indent=2))
    print("\nCurrent Config:")
    print(json.dumps(config.config, indent=2))


if __name__ == "__main__":
    main()