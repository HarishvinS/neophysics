"""
Train the T5 model for Neophysics.

This script loads a dataset, splits it into training and testing sets,
trains the model, and evaluates its performance on the test set.
"""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW
import numpy as np

class PhysicsDataset(Dataset):
    """PyTorch Dataset for physics command translation."""
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.texts = [item['text'] for item in data]
        self.actions = [item['action_sequence'] for item in data]
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        action = self.actions[idx]

        # T5 requires a prefix for the task
        input_text = f"translate English to Physics: {text}"

        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            action,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        # Replace padding token id in the labels with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten(),
            'source_text': text,
            'target_text': action
        }

def calculate_accuracy(model, tokenizer, dataloader, device):
    """Calculates exact match accuracy on a given dataset."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    mismatched_examples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            
            targets = batch['target_text']
            sources = batch['source_text']

            for i in range(len(preds)):
                pred_text = preds[i].strip()
                target_text = targets[i].strip()

                if pred_text == target_text:
                    correct_predictions += 1
                elif len(mismatched_examples) < 5: # Collect up to 5 examples
                    mismatched_examples.append({
                        "source": sources[i],
                        "target": target_text,
                        "predicted": pred_text
                    })
                total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, mismatched_examples

def train(args):
    """Main training and evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    dataset = PhysicsDataset(tokenizer, data, max_length=args.max_seq_length)

    # Split data into training and testing sets with a fixed seed for reproducibility
    dataset_size = len(dataset)
    if not 0 < args.test_split_size < 1:
        raise ValueError("test_split_size must be between 0 and 1.")
    test_size = int(np.floor(args.test_split_size * dataset_size))
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {len(train_dataset)} training, {len(test_dataset)} testing examples.")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    print("\nTraining finished. Running evaluation on the test set...")
    accuracy, mismatched = calculate_accuracy(model, tokenizer, test_dataloader, device)
    print(f"\nTest Set Accuracy (Exact Match): {accuracy:.4f}")

    if mismatched:
        print("\n--- Showing Mismatched Predictions from Test Set ---")
        for i, item in enumerate(mismatched):
            print(f"\nExample {i+1}:")
            print(f"  Source:    '{item['source']}'")
            print(f"  Target:    '{item['target']}'")
            print(f"  Predicted: '{item['predicted']}'")
            if item['target'].endswith(';') and not item['predicted'].endswith(';'):
                print("  Analysis:   Predicted sequence may be missing the trailing ';'")

    if args.output_dir:
        print(f"Saving model to {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a T5 model for Neophysics.")
    parser.add_argument("--data_path", type=str, default="data/training_data_balanced.json", help="Path to the training data JSON file.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Pre-trained model name.")
    parser.add_argument("--output_dir", type=str, default="models/physics_model", help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--test_split_size", type=float, default=0.2, help="Proportion of the dataset to use for testing (e.g., 0.2 for 20%).")
    
    args = parser.parse_args()
    train(args)