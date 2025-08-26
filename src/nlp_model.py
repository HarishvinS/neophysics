"""
Defines the sequence-to-sequence model architecture for the Learnable Physics Engine.
This model translates natural language commands into structured action sequences.
Now powered by OpenAI's gpt-oss-20b via Ollama for enhanced reasoning capabilities.
"""

import torch
import torch.nn as nn
import json
from openai import OpenAI
from typing import Optional


class Seq2SeqModel(nn.Module):
    """A wrapper around OpenAI's gpt-oss-20b model for physics command generation using Ollama."""

    def __init__(self, model_name="gpt-oss:20b"):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def _create_system_prompt(self) -> str:
        """Creates system prompt for physics command generation."""
        return """You are a physics simulation expert. Convert natural language commands into structured action sequences for a 3D physics engine.

Action Format:
CREATE id=obj1 type=sphere pos=(x,y,z) rot=(rx,ry,rz) scale=(sx,sy,sz) mass=m material=mat;
RELATE subject_id=obj1 type=on target_id=obj2;

Supported types: sphere, box, ramp
Supported materials: wood, metal, rubber, ice, plastic, glass, stone

Respond ONLY with the action sequence, no explanations."""
    
    def _create_user_prompt(self, text: str) -> str:
        """Creates user prompt for the specific command."""
        return f"Convert this command to action sequence: {text}"

    def generate(self, text: str, max_length: int = 512) -> str:
        """Generates an action sequence from a text prompt using Ollama via OpenAI API."""
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_length
            )
            
            result = response.choices[0].message.content
            return self._extract_action_sequence(result)
            
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return self._fallback_action()
    
    def _extract_action_sequence(self, response: str) -> str:
        """Extracts the action sequence from the model's response."""
        # Look for CREATE commands in the response
        lines = response.split('\n')
        action_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('CREATE') or line.startswith('RELATE'):
                action_lines.append(line)
        
        if action_lines:
            return ' '.join(action_lines)
        
        # Fallback: try to find any structured commands
        if 'CREATE' in response:
            # Extract everything after "Action sequence:" or similar
            for marker in ["Action sequence:", "Actions:", "CREATE"]:
                if marker in response:
                    idx = response.find(marker)
                    return response[idx:].strip()
        
        # Last resort: create a default sphere
        return "CREATE id=obj1 type=sphere pos=(0,0,1) rot=(0,0,0) scale=(0.2,0.2,0.2) mass=1.0 material=wood;"

    def _fallback_action(self) -> str:
        """Fallback action when Ollama fails."""
        return "CREATE id=obj1 type=sphere pos=(0,0,1) rot=(0,0,0) scale=(0.2,0.2,0.2) mass=1.0 material=wood;"
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass - not used for inference-only model."""
        raise NotImplementedError("Training not supported with Ollama backend")

    def save(self, save_directory: str):
        """Saves model configuration."""
        config = {"model_name": self.model_name}
        with open(f"{save_directory}/gpt_oss_config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, load_directory: str):
        """Loads model from configuration."""
        try:
            with open(f"{load_directory}/gpt_oss_config.json", "r") as f:
                config = json.load(f)
            return cls(model_name=config["model_name"])
        except FileNotFoundError:
            return cls()


# LoRA fine-tuned GPT-OSS-20B model
class GPTOSSLoRAModel(nn.Module):
    """LoRA fine-tuned GPT-OSS-20B model for physics command generation."""
    
    def __init__(self, base_model="openai/gpt-oss-20b", lora_path="models/gpt_oss_lora"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        self.base_model = base_model
        self.lora_path = lora_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Quantization config
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load base GPT-OSS-20B model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded GPT-OSS-20B LoRA adapter from {lora_path}")
        except:
            print(f"No LoRA adapter found at {lora_path}, using base GPT-OSS-20B")
    
    def generate(self, text: str, max_length: int = 512) -> str:
        """Generate action sequence from natural language."""
        # Create prompt
        system_msg = "You are a physics simulation expert. Convert natural language commands into structured action sequences for a 3D physics engine.\n\nAction Format:\nCREATE id=obj1 type=sphere pos=(x,y,z) rot=(rx,ry,rz) scale=(sx,sy,sz) mass=m material=mat;\nRELATE subject_id=obj1 type=on target_id=obj2;\n\nSupported types: sphere, box, ramp\nSupported materials: wood, metal, rubber, ice, plastic, glass, stone\n\nRespond ONLY with the action sequence, no explanations."
        user_msg = f"Convert this command to action sequence: {text}"
        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._extract_action_sequence(response)
    
    def _extract_action_sequence(self, response: str) -> str:
        """Extract action sequence from response."""
        lines = response.split('\n')
        action_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('CREATE') or line.startswith('RELATE'):
                action_lines.append(line)
        
        if action_lines:
            return ' '.join(action_lines)
        
        # Fallback
        if 'CREATE' in response:
            for marker in ["Action sequence:", "Actions:", "CREATE"]:
                if marker in response:
                    idx = response.find(marker)
                    return response[idx:].strip()
        
        return "CREATE id=obj1 type=sphere pos=(0,0,1) rot=(0,0,0) scale=(0.2,0.2,0.2) mass=1.0 material=wood;"


# Legacy T5 model for backward compatibility
class T5Seq2SeqModel(nn.Module):
    """Legacy T5 model wrapper for backward compatibility."""
    
    def __init__(self, model_name="t5-small"):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def generate(self, text: str, max_length: int = 256) -> str:
        """Generate using T5 model."""
        self.model.eval()
        input_text = f"translate English to ActionSequence: {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
        
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    @classmethod
    def load(cls, load_directory: str):
        """Load T5 model from directory."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        instance = cls.__new__(cls)
        super(T5Seq2SeqModel, instance).__init__()
        instance.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        instance.tokenizer = T5Tokenizer.from_pretrained(load_directory)
        return instance