"""
Defines the sequence-to-sequence model architecture for the Learnable Physics Engine.
This model translates natural language commands into structured action sequences.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Seq2SeqModel(nn.Module):
    """A wrapper around a pre-trained T5 model for sequence-to-sequence tasks."""

    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, text: str, max_length: int = 256) -> str:
        """Generates an action sequence from a text prompt for inference."""
        self.model.eval()
        # T5 requires a prefix for text-to-text tasks.
        input_text = f"translate English to ActionSequence: {text}"
        
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
        
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,  # Use beam search for better results
                early_stopping=True
            )
        
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    def save(self, save_directory: str):
        """Saves the model and tokenizer to a directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def load(cls, load_directory: str):
        """Loads a model and tokenizer from a directory."""
        instance = cls.__new__(cls)
        super(Seq2SeqModel, instance).__init__()
        instance.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        instance.tokenizer = T5Tokenizer.from_pretrained(load_directory)
        return instance

