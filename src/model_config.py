"""
Model configuration utility for Neophysics.
Allows users to choose between different model backends based on their system capabilities.
"""

import os
import json
import psutil
from typing import Dict, Any


class ModelConfig:
    """Configuration manager for selecting the appropriate model backend."""
    
    def __init__(self):
        self.config_path = "models/model_config.json"
        self.config = self.load_config()
    
    def detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system RAM, GPU availability, and recommend model backend."""
        capabilities = {
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 1),
            "has_cuda": False,
            "recommended_backend": "t5-small"
        }
        
        # Check for CUDA
        try:
            import torch
            capabilities["has_cuda"] = torch.cuda.is_available()
            if capabilities["has_cuda"]:
                capabilities["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        except ImportError:
            pass
        
        # Recommend backend based on capabilities
        if capabilities["available_ram_gb"] >= 16 and capabilities["has_cuda"]:
            capabilities["recommended_backend"] = "gpt-oss-20b"
        elif capabilities["available_ram_gb"] >= 8:
            capabilities["recommended_backend"] = "gpt-oss-20b-cpu"
        else:
            capabilities["recommended_backend"] = "t5-small"
        
        return capabilities
    
    def get_model_config(self, backend: str = None) -> Dict[str, Any]:
        """Get configuration for specified backend or auto-detect."""
        if backend is None:
            capabilities = self.detect_system_capabilities()
            backend = capabilities["recommended_backend"]
        
        configs = {
            "gpt-oss-20b": {
                "model_name": "openai/gpt-oss-20b",
                "use_vllm": True,
                "memory_requirement_gb": 16,
                "description": "Full gpt-oss-20b with vLLM for optimal performance"
            },
            "gpt-oss-20b-cpu": {
                "model_name": "openai/gpt-oss-20b",
                "use_vllm": False,
                "memory_requirement_gb": 8,
                "description": "gpt-oss-20b with standard transformers (slower but less memory)"
            },
            "t5-small": {
                "model_name": "t5-small",
                "use_vllm": False,
                "memory_requirement_gb": 2,
                "description": "Lightweight T5 model for resource-constrained systems"
            }
        }
        
        return configs.get(backend, configs["t5-small"])
    
    def save_config(self, backend: str):
        """Save the selected backend configuration."""
        config = self.get_model_config(backend)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, "w") as f:
            json.dump({"selected_backend": backend, **config}, f, indent=2)
        
        self.config = config
    
    def load_config(self) -> Dict[str, Any]:
        """Load saved configuration or return default."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_model_config("t5-small")
    
    def interactive_setup(self):
        """Interactive setup for model selection."""
        print("Neophysics Model Configuration")
        print("=" * 40)
        
        capabilities = self.detect_system_capabilities()
        print(f"System RAM: {capabilities['total_ram_gb']} GB")
        print(f"Available RAM: {capabilities['available_ram_gb']} GB")
        print(f"CUDA Available: {capabilities['has_cuda']}")
        if capabilities['has_cuda']:
            print(f"GPU Memory: {capabilities.get('gpu_memory_gb', 'Unknown')} GB")
        
        print(f"\nRecommended: {capabilities['recommended_backend']}")
        print("\nAvailable backends:")
        
        backends = ["gpt-oss-20b", "gpt-oss-20b-cpu", "t5-small"]
        for i, backend in enumerate(backends, 1):
            config = self.get_model_config(backend)
            status = "[OK]" if capabilities['available_ram_gb'] >= config['memory_requirement_gb'] else "[INSUFFICIENT RAM]"
            print(f"{i}. {backend} {status}")
            print(f"   {config['description']}")
            print(f"   Memory required: {config['memory_requirement_gb']} GB")
        
        while True:
            try:
                choice = input(f"\nSelect backend (1-{len(backends)}) or press Enter for recommended: ").strip()
                if not choice:
                    selected = capabilities['recommended_backend']
                    break
                else:
                    selected = backends[int(choice) - 1]
                    break
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")
        
        self.save_config(selected)
        print(f"\nConfiguration saved: {selected}")
        return selected


def main():
    """Run interactive model configuration."""
    config = ModelConfig()
    config.interactive_setup()


if __name__ == "__main__":
    main()