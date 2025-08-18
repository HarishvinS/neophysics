"""
Week 3 Demo - Basic ML Pipeline
Demonstrates the completed Week 3 functionality: neural network training and evaluation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import json
from model_architecture import TextToSceneModel, ModelConfig
from training_pipeline import create_training_pipeline
from evaluation_system import ModelEvaluator
from scene_encoder import DatasetProcessor


def demo_model_architecture():
    """Demonstrate the neural network architecture."""
    print("🧠 Neural Network Architecture Demo")
    print("=" * 40)
    
    # Create model
    config = ModelConfig()
    model = TextToSceneModel(
        hidden_size=config.hidden_size,
        max_objects=config.max_objects
    )
    
    print(f"Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Max objects: {config.max_objects}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    test_texts = [
        "create a ball on a ramp",
        "add two spheres",
        "place a box"
    ]
    
    print(f"\nTesting forward pass with {len(test_texts)} examples...")
    
    with torch.no_grad():
        predictions = model(test_texts)
    
    print("Output tensor shapes:")
    for key, tensor in predictions.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test scene prediction
    print(f"\nTesting scene prediction...")
    predicted_scene = model.predict_scene("create a ball on a ramp")
    objects = [obj for obj in predicted_scene.objects if obj.object_type.value != 'plane']
    
    print(f"Predicted scene with {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  {i+1}. {obj.object_type.value} ({obj.material.value}) at {obj.position.to_list()}")


def demo_training_pipeline():
    """Demonstrate the training pipeline."""
    print("\n🏋️ Training Pipeline Demo")
    print("=" * 30)
    
    # Check for dataset
    dataset_path = "data/physics_dataset_full.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: python src/generate_dataset.py --num_examples 100 --quick")
        return
    
    # Create small config for demo
    config = ModelConfig()
    config.num_epochs = 1  # Just one epoch for demo
    config.batch_size = 4
    
    print(f"Demo training configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create training pipeline
    print(f"\nSetting up training pipeline...")
    trainer = create_training_pipeline(dataset_path, config)
    
    print(f"Training data loaded:")
    print(f"  Train batches: {len(trainer.train_loader)}")
    print(f"  Validation batches: {len(trainer.val_loader) if trainer.val_loader else 0}")
    
    # Run one training epoch
    print(f"\nRunning one training epoch...")
    train_losses = trainer.train_epoch()
    
    print(f"Training losses:")
    for key, loss in train_losses.items():
        print(f"  {key}: {loss:.4f}")
    
    # Run validation
    if trainer.val_loader:
        print(f"\nRunning validation...")
        val_losses = trainer.validate()
        print(f"Validation losses:")
        for key, loss in val_losses.items():
            print(f"  {key}: {loss:.4f}")


def demo_evaluation_system():
    """Demonstrate the evaluation system."""
    print("\n🔍 Evaluation System Demo")
    print("=" * 30)
    
    # Create model
    config = ModelConfig()
    model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, max_objects=config.max_objects)
    
    # Test evaluation on sample texts
    test_texts = [
        "create a ball",
        "add a ramp and a sphere",
        "place two boxes next to each other"
    ]
    
    print(f"Evaluating model on {len(test_texts)} examples...")
    results = evaluator.evaluate_text_examples(test_texts)
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}: '{result['text']}'")
        print(f"  Predicted objects: {result['num_objects']}")
        print(f"  Physics plausibility: {result['physics_score']['overall']:.3f}")
        
        # Show breakdown
        physics_breakdown = result['physics_score']
        print(f"  Plausibility breakdown:")
        for key, score in physics_breakdown.items():
            if key != 'overall':
                print(f"    {key}: {score:.3f}")


def demo_end_to_end():
    """Demonstrate end-to-end text-to-scene translation."""
    print("\n🎯 End-to-End Demo")
    print("=" * 20)
    
    # Check if trained model exists
    model_path = "models/trained_model/final_model.pth"
    
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        
        # Load model
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Pre-trained model loaded successfully!")
        
        # Test with various inputs
        demo_inputs = [
            "create a red ball",
            "add a wooden ramp",
            "place a sphere on a ramp",
            "create two metal boxes",
            "add a bouncy rubber ball"
        ]
        
        print(f"\nTesting {len(demo_inputs)} text inputs:")
        
        with torch.no_grad():
            for i, text in enumerate(demo_inputs):
                print(f"\n{i+1}. Input: '{text}'")
                
                # Get prediction
                predicted_scene = model.predict_scene(text)
                
                # Show results
                objects = [obj for obj in predicted_scene.objects if obj.object_type.value != 'plane']
                print(f"   Output: {len(objects)} objects predicted")
                
                for j, obj in enumerate(objects[:2]):  # Show first 2 objects
                    print(f"     {j+1}. {obj.object_type.value} ({obj.material.value})")
                    print(f"        Position: ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f})")
                    print(f"        Mass: {obj.mass:.2f}kg")
    
    else:
        print("No pre-trained model found.")
        print("To train a model, run:")
        print("  python src/train_model.py --quick")
        
        # Show what the pipeline would do
        print("\nDemonstrating untrained model behavior...")
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        with torch.no_grad():
            predicted_scene = model.predict_scene("create a ball on a ramp")
            objects = [obj for obj in predicted_scene.objects if obj.object_type.value != 'plane']
            print(f"Untrained model predicts {len(objects)} objects (random initialization)")


def demo_training_results():
    """Show training results if available."""
    print("\n📊 Training Results Demo")
    print("=" * 25)
    
    # Check for training history
    history_path = "models/trained_model/training_history.json"
    eval_path = "models/trained_model/evaluation_results.json"
    
    if os.path.exists(history_path):
        print("Loading training history...")
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print(f"Training Summary:")
        print(f"  Final epoch: {history['final_epoch']}")
        print(f"  Training time: {history['training_time']:.2f} seconds")
        print(f"  Final train loss: {history['train_losses'][-1]['total']:.4f}")
        if history['val_losses']:
            print(f"  Final val loss: {history['val_losses'][-1]['total']:.4f}")
    
    if os.path.exists(eval_path):
        print("\nLoading evaluation results...")
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
        
        dataset_eval = eval_results['dataset_evaluation']
        scene_metrics = dataset_eval['scene_metrics']
        physics_metrics = dataset_eval['physics_metrics']
        
        print(f"Model Performance:")
        print(f"  Object count accuracy: {scene_metrics['object_count_accuracy']:.3f}")
        print(f"  Material accuracy: {scene_metrics['material_accuracy']:.3f}")
        print(f"  Physics plausibility: {physics_metrics['overall']:.3f}")
        print(f"  Test examples: {eval_results['test_set_size']}")
    
    if not os.path.exists(history_path) and not os.path.exists(eval_path):
        print("No training results found.")
        print("To train and evaluate a model, run:")
        print("  python src/train_model.py --quick")


def main():
    """Run the complete Week 3 demo."""
    print("🎬 Week 3 Demo: Basic ML Pipeline")
    print("=" * 50)
    print("Demonstrating neural network training and evaluation capabilities")
    print("=" * 50)
    
    # Run all demos
    demo_model_architecture()
    demo_training_pipeline()
    demo_evaluation_system()
    demo_end_to_end()
    demo_training_results()
    
    print("\n" + "=" * 50)
    print("🎉 Week 3 Demo Complete!")
    print("=" * 50)
    
    print("\nKey Achievements:")
    print("✅ Neural network architecture (67M parameters)")
    print("✅ Text encoder using pre-trained transformers")
    print("✅ Scene decoder with multi-task outputs")
    print("✅ Complete training pipeline with loss functions")
    print("✅ Comprehensive evaluation system")
    print("✅ Physics plausibility assessment")
    print("✅ End-to-end text-to-scene translation")
    
    print("\nTechnical Highlights:")
    print("🧠 DistilBERT-based text encoder")
    print("🎯 Multi-task learning (count, type, properties)")
    print("📊 Custom loss functions for physics scenes")
    print("🔍 Automated evaluation metrics")
    print("⚖️ Physics validation scoring")
    
    print("\nNext Steps:")
    print("🚀 Ready for Week 4: Scene-to-Physics Integration")
    print("🔗 Connect ML predictions to PyBullet simulation")
    print("🎮 Real-time physics validation")
    print("🔄 Continuous learning from simulation outcomes")


if __name__ == "__main__":
    main()
