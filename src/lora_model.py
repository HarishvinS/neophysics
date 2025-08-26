"""
LoRA fine-tuned model wrapper for physics command generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
from peft import PeftModel
import json


class LoRASeq2SeqModel:
    """Wrapper for LoRA fine-tuned GPT-OSS-20B model."""
    
    def __init__(self, base_model="openai/gpt-oss-20b", lora_path="models/gpt_oss_lora"):
        self.base_model = base_model
        self.lora_path = lora_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Configure device mapping based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_map = {"":0} if gpu_memory >= 20 else "balanced_low_0"
        else:
            device_map = "cpu"
        
        # Load base GPT-OSS-20B model with optional quantization
        try:
            if BitsAndBytesConfig and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                raise ImportError("BitsAndBytesConfig not available")
        except Exception as e:
            print(f"Loading without quantization: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
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
        messages = [
            {
                "role": "system",
                "content": "You are a physics simulation expert. Convert natural language commands into structured action sequences for a 3D physics engine.\n\nAction Format:\nCREATE id=obj1 type=sphere pos=(x,y,z) rot=(rx,ry,rz) scale=(sx,sy,sz) mass=m material=mat;\nRELATE subject_id=obj1 type=on target_id=obj2;\n\nSupported types: sphere, box, ramp\nSupported materials: wood, metal, rubber, ice, plastic, glass, stone\n\nRespond ONLY with the action sequence, no explanations."
            },
            {
                "role": "user",
                "content": f"Convert this command to action sequence: {text}"
            }
        ]
        
        # Create prompt manually since GPT-OSS may not have chat template
        prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        elif torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
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