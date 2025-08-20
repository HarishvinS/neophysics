"""
Complete Model Training Script
Trains the text-to-scene model end-to-end and validates performance.
"""

import torch
import os
import numpy as np
import json
import argparse
import time
from typing import Dict, List
from torch.utils.data import DataLoader, Subset

from model_architecture import TextToSceneModel, ModelConfig
from training_pipeline import Trainer
from evaluation_system import ModelEvaluator
from scene_encoder import SceneEncoder, PhysicsDataset


def train_model(dataset_path: str, config: ModelConfig, save_dir: str = "models"):
    """
    Train the complete text-to-scene model.
    
    Args:
        dataset_path: Path to training dataset
        config: Model configuration
        save_dir: Directory to save models and results
        
    Returns:
        Trained model and evaluation results
    """
    print("🚀 Starting Text-to-Scene Model Training")
    print("=" * 50)
    
    # Sanitize and create save directory
    save_dir = os.path.normpath(save_dir)
    if '..' in save_dir or save_dir.startswith('/'):
        raise ValueError("Invalid save directory: path traversal detected")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # 1. Create the full dataset
    # Encoder is no longer needed here, the model's tokenizer handles it.
    full_dataset = PhysicsDataset(dataset_path=dataset_path)
    
    # 2. Create shuffled indices for splitting
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # 3. Define split points and create subsets
    train_split = int(np.floor(0.7 * dataset_size))
    val_split = int(np.floor(0.85 * dataset_size))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    print(f"\nDataset split:")
    print(f"  Training examples: {len(train_subset)}")
    print(f"  Validation examples: {len(val_subset)}")
    print(f"  Test examples: {len(test_subset)}")
    
    # Create DataLoaders
    collate_fn = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model and trainer
    model = TextToSceneModel(config=config)
    trainer = Trainer(config, model, train_loader, val_loader)

    print("\n🏋️ Training model...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path, is_best=True)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'training_time': training_time,
        'final_epoch': trainer.epoch
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return trainer.model, history, test_subset


def evaluate_model(model: TextToSceneModel, test_dataset: Subset, config: ModelConfig) -> Dict:
    """
    Evaluate the trained model comprehensively.
    
    Args:
        model: Trained model
        dataset_path: Path to evaluation dataset
        config: Model configuration

    Returns:
        Evaluation results
    """
    print("\n🔍 Evaluating model on test set...")
    model.eval()
    
    # For seq2seq, we can use metrics like BLEU/ROUGE, or a simpler custom metric.
    # Here, we'll use a simple exact match and semantic similarity score.
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
    
    total_examples = 0
    exact_matches = 0
    
    with torch.no_grad():
        for texts, targets in test_loader:
            for i in range(len(texts)):
                text = texts[i]
                target_sequence = targets[i]
                predicted_sequence = model.predict_action_sequence(text)
                
                if predicted_sequence.strip() == target_sequence.strip():
                    exact_matches += 1
                total_examples += 1

    accuracy = exact_matches / total_examples if total_examples > 0 else 0
    
    return {
        'test_set_size': total_examples,
        'exact_match_accuracy': accuracy
    }
    
    return full_results


def print_evaluation_summary(results: Dict):
    """Print a summary of evaluation results."""
    print("\n📈 Evaluation Summary")
    print("=" * 30)
    
    if 'exact_match_accuracy' in results:
        print(f"Exact Match Accuracy on Test Set: {results['exact_match_accuracy']:.2%}")
    else:
        print("Evaluation results format not recognized.")


def demonstrate_model(model: TextToSceneModel):
    """Demonstrate the trained model with interactive examples."""
    print("\n🎭 Model Demonstration")
    print("=" * 25)
    
    # Test examples
    demo_texts = [
        "create a red ball",
        "add a wooden ramp",
        "place a sphere on a ramp so it rolls down",
        "create two boxes next to each other",
        "add a bouncy ball that will bounce high"
    ]
    
    model.eval()
    with torch.no_grad():
        for text in demo_texts:
            print(f"\nInput: '{text}'")
            action_sequence = model.predict_action_sequence(text)
            print(f"  > Predicted Action Sequence: {action_sequence}")


def main():
    """Main training and evaluation function."""
    parser = argparse.ArgumentParser(description='Train text-to-scene model')
    parser.add_argument('--dataset', type=str, default='data/physics_dataset_full.json',
                       help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models/trained_model',
                       help='Directory to save model and results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with fewer epochs')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        print("Please generate a dataset first:")
        print("  python src/generate_dataset.py --num_examples 1000")
        return
    
    # Create configuration
    config = ModelConfig()
    config.num_epochs = 5 if args.quick else args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    if args.quick:
        print("🚀 Quick training mode enabled")
    
    print(f"Training Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Save directory: {args.save_dir}")
    
    try:
        # Train model
        model, training_history, test_dataset = train_model(args.dataset, config, args.save_dir)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_dataset, config)
        
        # Save evaluation results
        eval_path = os.path.join(args.save_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(eval_results, default=str))
            json.dump(serializable_results, f, indent=2)
        print(f"Evaluation results saved to {eval_path}")
        
        # Print summary
        print_evaluation_summary(eval_results)
        
        # Demonstrate model
        demonstrate_model(model)
        
        print(f"\n🎉 Training and evaluation completed successfully!")
        print(f"Model saved in: {args.save_dir}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
