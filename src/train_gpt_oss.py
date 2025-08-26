"""
Fine-tune GPT-OSS-20B using LoRA for physics command generation.
Uses PEFT for efficient LoRA training on GPT-OSS-20B.
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import argparse


def load_physics_data(data_path: str):
    """Load and format physics training data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Format as chat conversation
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a physics simulation expert. Convert natural language commands into structured action sequences for a 3D physics engine.\n\nAction Format:\nCREATE id=obj1 type=sphere pos=(x,y,z) rot=(rx,ry,rz) scale=(sx,sy,sz) mass=m material=mat;\nRELATE subject_id=obj1 type=on target_id=obj2;\n\nSupported types: sphere, box, ramp\nSupported materials: wood, metal, rubber, ice, plastic, glass, stone\n\nRespond ONLY with the action sequence, no explanations."
                },
                {
                    "role": "user", 
                    "content": f"Convert this command to action sequence: {item['input']}"
                },
                {
                    "role": "assistant",
                    "content": item['output']
                }
            ]
        }
        formatted_data.append(conversation)
    
    return Dataset.from_list(formatted_data)


def format_chat_template(example, tokenizer):
    """Format conversation using chat template."""
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/training_data.json")
    parser.add_argument("--output_dir", default="models/gpt_oss_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    # Load GPT-OSS-20B model and tokenizer
    model_name = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check available GPU memory and configure device mapping
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"Available GPU memory: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 20:  # Enough for full model
            device_map = {"":0}
        else:  # Use CPU offloading
            device_map = "balanced_low_0"
    else:
        device_map = "cpu"
    
    # Try with quantization, fallback without if it fails
    try:
        if BitsAndBytesConfig and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            raise ImportError("Quantization not available")
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Loading model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # GPT-OSS attention modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format data
    dataset = load_physics_data(args.data_path)
    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch" if not torch.cuda.is_available() else "adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        dataloader_drop_last=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    # Train
    print("Starting LoRA fine-tuning on GPT-OSS-20B...")
    trainer.train()

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()