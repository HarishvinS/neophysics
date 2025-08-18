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
from scene_encoder import SceneEncoder, DatasetProcessor


class MultiTaskLoss(nn.Module):
    """Multi-task loss function for text-to-scene translation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize multi-task loss."""
        super().__init__()
        
        self.config = config
        
        # Loss functions for different components
        self.object_count_loss = nn.CrossEntropyLoss()
        self.object_type_loss = nn.BCEWithLogitsLoss()
        self.position_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()
        self.scale_loss = nn.MSELoss()
        self.mass_loss = nn.MSELoss()
        self.material_loss = nn.CrossEntropyLoss()
        self.gravity_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Object count loss
        object_count_pred = predictions['object_count_probs']
        object_count_target = torch.argmax(targets['object_count'], dim=1)
        losses['object_count'] = self.object_count_loss(object_count_pred, object_count_target)
        
        # Object type loss (multi-label) - use logits for BCEWithLogitsLoss
        object_type_logits = predictions['object_type_logits']
        object_type_target = targets['object_types']
        losses['object_type'] = self.object_type_loss(object_type_logits, object_type_target)
        
        # Object properties losses
        losses['position'] = self.position_loss(
            predictions['object_positions'], targets['object_positions']
        )
        losses['rotation'] = self.rotation_loss(
            predictions['object_rotations'], targets['object_rotations']
        )
        losses['scale'] = self.scale_loss(
            predictions['object_scales'], targets['object_scales']
        )
        losses['mass'] = self.mass_loss(
            predictions['object_masses'], targets['object_masses']
        )
        
        # Material loss
        material_pred_flat = predictions['object_material_probs'].view(-1, predictions['object_material_probs'].shape[-1])
        material_target_flat = torch.argmax(targets['object_materials'].view(-1, targets['object_materials'].shape[-1]), dim=1)
        losses['material'] = self.material_loss(material_pred_flat, material_target_flat)
        
        # Environment loss
        losses['gravity'] = self.gravity_loss(predictions['gravity'], targets['gravity'])
        
        # Weighted total loss
        props_loss = losses['position'] + losses['rotation'] + losses['scale'] + losses['mass'] + losses['material']
        total_loss = (
            self.config.object_count_weight * losses['object_count'] +
            self.config.object_type_weight * losses['object_type'] +
            self.config.object_props_weight * props_loss +
            self.config.environment_weight * losses['gravity']
        )
        
        losses['total'] = total_loss
        
        return losses


class Trainer:
    """Main training class for the text-to-scene model."""
    
    def __init__(self, config: ModelConfig, model: TextToSceneModel, 
                 train_loader: DataLoader, val_loader: DataLoader = None,
                 texts: List[str] = None):
        """
        Initialize trainer.
        
        Args:
            config: Model configuration
            model: Text-to-scene model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            texts: List of text descriptions for data loader indices
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.texts = texts or []
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
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
        
        epoch_losses = {
            'total': 0.0, 'object_count': 0.0, 'object_type': 0.0,
            'position': 0.0, 'rotation': 0.0, 'scale': 0.0,
            'mass': 0.0, 'material': 0.0, 'gravity': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Extract batch data
            text_indices = batch[0]
            targets = {
                'object_count': batch[1],
                'object_types': batch[2],
                'object_positions': batch[3],
                'object_rotations': batch[4],
                'object_scales': batch[5],
                'object_masses': batch[6],
                'object_materials': batch[7],
                'gravity': batch[8]
            }
            
            # Get texts for this batch
            batch_texts = [self.texts[idx] for idx in text_indices]
            
            # Forward pass
            predictions = self.model(batch_texts)
            
            # Compute loss
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()
            
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {losses['total'].item():.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_losses = {
            'total': 0.0, 'object_count': 0.0, 'object_type': 0.0,
            'position': 0.0, 'rotation': 0.0, 'scale': 0.0,
            'mass': 0.0, 'material': 0.0, 'gravity': 0.0
        }
        
        num_batches = 0
        
        with torch.inference_mode():
            for batch in self.val_loader:
                # Extract batch data
                text_indices = batch[0]
                targets = {
                    'object_count': batch[1],
                    'object_types': batch[2],
                    'object_positions': batch[3],
                    'object_rotations': batch[4],
                    'object_scales': batch[5],
                    'object_masses': batch[6],
                    'object_materials': batch[7],
                    'gravity': batch[8]
                }
                
                # Get texts for this batch
                batch_texts = [self.texts[idx] for idx in text_indices]
                
                # Forward pass
                predictions = self.model(batch_texts)
                
                # Compute loss
                losses = self.criterion(predictions, targets)
                
                # Accumulate losses
                for key, loss in losses.items():
                    val_losses[key] += loss.item()
                
                num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        return val_losses
    
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


def create_training_pipeline(dataset_path: str, config: ModelConfig = None) -> Trainer:
    """
    Create a complete training pipeline.
    
    Args:
        dataset_path: Path to training dataset
        config: Model configuration (uses default if None)
        
    Returns:
        Configured trainer
    """
    if config is None:
        config = ModelConfig()
    
    print("Setting up training pipeline...")
    
    # Load and prepare data
    processor = DatasetProcessor(max_objects=config.max_objects)
    texts, encoded_scenes = processor.prepare_training_data(dataset_path)
    
    # Shuffle data before splitting to prevent bias
    import random
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    # Split into train/validation
    split_idx = int(0.8 * len(texts))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    
    train_encoded = {key: tensor[train_indices] for key, tensor in encoded_scenes.items()}
    val_encoded = {key: tensor[val_indices] for key, tensor in encoded_scenes.items()}
    
    print(f"Train examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    
    # Create data loaders
    train_loader, _ = processor.create_data_loader(
        train_texts, train_encoded, batch_size=config.batch_size, shuffle=True
    )
    val_loader, _ = processor.create_data_loader(
        val_texts, val_encoded, batch_size=config.batch_size, shuffle=False
    )
    
    # Create model with configuration
    model = TextToSceneModel(
        hidden_size=config.hidden_size,
        max_objects=config.max_objects,
        config=config
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        texts=texts  # Full text list for index lookup
    )
    
    return trainer


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
