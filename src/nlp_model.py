"""
Defines the sequence-to-sequence model architecture for the Learnable Physics Engine.
This model translates natural language commands into structured action sequences.
Powered by T5.
"""

import torch
import torch.nn as nn
import json
from typing import Optional

class PhysicsTranslationModel(nn.Module):
    """T5 model wrapper for physics command generation."""
    
    def __init__(self, model_name="t5-small"):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    
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
        return cls(model_name=load_directory)