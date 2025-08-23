"""
Training script for the Learnable Physics Engine's sequence-to-sequence model.

This script handles:
1. Generating or loading training data.
2. Setting up a PyTorch Dataset and DataLoader.
3. Initializing the T5 model.
4. Running the training loop.
5. Saving the trained model for inference.
"""

import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm

from nlp_model import Seq2SeqModel
from generate_dataset import RobustDataGenerator, summarize_dataset



# A placeholder for dynamic_scene_representation if it's not available when running standalone
try:
    from dynamic_scene_representation import ObjectType, MaterialType, Vector3, DynamicPhysicsScene, DynamicPhysicsObject
except ImportError:
    print("Warning: `dynamic_scene_representation` not found. Using placeholders for training script.")
    class ObjectType:
        SPHERE = "sphere"
        BOX = "box"
        RAMP = "ramp"
    class MaterialType:
        WOOD = "wood"
        METAL = "metal"
        RUBBER = "rubber"
    class Vector3:
        def __init__(self, x, y, z): pass
        def to_tuple(self): return (0, 0, 0)
    class DynamicPhysicsObject: pass
    class DynamicPhysicsScene: pass


class PhysicsDataset(Dataset):
    """PyTorch Dataset for text-to-action-sequence data."""
    def __init__(self, data, tokenizer, source_max_len=128, target_max_len=256):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.prefix = "translate English to ActionSequence: "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        source_text = self.prefix + item['text']
        target_text = item['action_sequence']

        source = self.tokenizer(source_text, max_length=self.source_max_len, padding='max_length', truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=self.target_max_len, padding='max_length', truncation=True, return_tensors="pt")

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        
        # Replace padding token id in the labels with -100 for loss calculation
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": source_ids, "attention_mask": source_mask, "labels": labels}

def train(model, train_loader, optimizer, scheduler, device, epochs=3):
    """The main training loop."""
    model.to(device)
    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train the sequence-to-sequence model for the physics engine.")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--num_examples', type=int, default=500, help='Number of examples to generate if data file does not exist.')
    parser.add_argument('--data_path', type=str, default="training_data.json", help='Path to the training data JSON file.')
    parser.add_argument('--save_path', type=str, default="models/physics_model", help='Directory to save the trained model.')
    
    args = parser.parse_args()

    # --- Configuration ---
    TRAIN_DATA_PATH = args.data_path
    MODEL_SAVE_PATH = args.save_path
    NUM_EXAMPLES = args.num_examples
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs

    # --- 1. Generate Data ---
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"'{TRAIN_DATA_PATH}' not found. Generating new dataset...")
        generator = RobustDataGenerator()
        dataset = generator.generate_dataset(num_examples=NUM_EXAMPLES)
        if dataset:
            summarize_dataset(dataset)
    else:
        print(f"Found existing dataset at '{TRAIN_DATA_PATH}'.")
        with open(TRAIN_DATA_PATH, 'r') as f:
            dataset = json.load(f)

    # --- 2. Initialize Model and Tokenizer ---
    print("Initializing T5 model and tokenizer...")
    model = Seq2SeqModel(model_name="t5-small")
    
    # --- 3. Create Dataset and DataLoader ---
    train_dataset = PhysicsDataset(dataset, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. Setup Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # --- 5. Train the Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")
    train(model, train_loader, optimizer, scheduler, device, epochs=EPOCHS)

    # --- 6. Save the Model ---
    print(f"Training complete. Saving model to '{MODEL_SAVE_PATH}'...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    main()