"""
Training Pipeline for Text-to-Scene Model
Implements training loop, loss functions, optimization, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional

from model_architecture import TextToSceneModel, ModelConfig


class Trainer:
    """Main training class for the text-to-scene model."""
    
    def __init__(self, config: ModelConfig, model: TextToSceneModel, 
                 train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Initialize trainer.
        
        Args:
            config: Model configuration
            model: Text-to-scene model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create checkpoint directory
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', 'models/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (texts, target_sequences) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            # The model now handles tokenization and loss calculation internally
            outputs = self.model(texts=list(texts), targets=list(target_sequences))
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'total': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.inference_mode():
            for texts, target_sequences in self.val_loader:
                
                # Forward pass
                outputs = self.model(texts=list(texts), targets=list(target_sequences))
                loss = outputs.loss
                total_loss += loss.item()
                
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'total': avg_loss}
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        # Sanitize filepath
        safe_filepath = os.path.abspath(filepath)
        if not safe_filepath.startswith(os.path.abspath('models')):
            raise ValueError("Checkpoint path must be within models directory")
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__
        }
        
        try:
            torch.save(checkpoint, safe_filepath)
            if is_best:
                best_path = safe_filepath.replace('.pth', '_best.pth')
                torch.save(checkpoint, best_path)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        # Sanitize filepath
        safe_filepath = os.path.abspath(filepath)
        if not safe_filepath.startswith(os.path.abspath('models')):
            raise ValueError("Checkpoint path must be within models directory")
        
        try:
            checkpoint = torch.load(safe_filepath)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
        except (FileNotFoundError, IOError, OSError) as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Train with error handling
            try:
                train_losses = self.train_epoch()
                self.train_losses.append(train_losses)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA out of memory error. Try reducing batch size.")
                    raise
                else:
                    raise
            
            # Validate
            try:
                val_losses = self.validate()
            except Exception as e:
                print(f"Validation error: {e}")
                val_losses = {}
            if val_losses:
                self.val_losses.append(val_losses)
                
                # Learning rate scheduling
                self.scheduler.step(val_losses['total'])
                
                # Check for best model
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
            else:
                is_best = False
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            if val_losses:
                print(f"  Val Loss: {val_losses['total']:.4f}")
                if is_best:
                    print("  *** New best model! ***")
            
            # Save checkpoint (including final epoch)
            if (epoch + 1) % self.config.save_frequency == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save_checkpoint(checkpoint_path, is_best)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")


def test_training_pipeline():
    """Test the training pipeline with a small dataset."""
    print("Testing training pipeline...")
    
    # Check if dataset exists
    dataset_path = "data/physics_dataset_full.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: python src/generate_dataset.py --num_examples 100 --quick")
        return
    
    # Create small config for testing
    config = ModelConfig()
    config.num_epochs = 2
    config.batch_size = 4
    config.learning_rate = 1e-3
    
    # Create trainer
    trainer = create_training_pipeline(dataset_path, config)
    
    # Test one epoch
    print("\nTesting one training epoch...")
    train_losses = trainer.train_epoch()
    print(f"Train losses: {train_losses}")
    
    # Test validation
    print("\nTesting validation...")
    val_losses = trainer.validate()
    print(f"Validation losses: {val_losses}")
    
    # Test checkpoint saving
    print("\nTesting checkpoint saving...")
    trainer.save_checkpoint("models/test_checkpoint.pth")
    print("Checkpoint saved successfully")
    
    print("\n✅ Training pipeline test completed!")


if __name__ == "__main__":
    test_training_pipeline()
