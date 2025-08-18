"""
Dataset Generation Script
Generates a large, high-quality training dataset for the physics engine.
"""

import os
import json
import time
from typing import List
import argparse

from data_generator import DataGenerator
from data_validator import DatasetValidator
from scene_representation import TrainingExample


def generate_filtered_dataset(num_examples: int, min_score: float = 0.5, 
                             max_attempts: int = None) -> List[TrainingExample]:
    """
    Generate a filtered dataset with quality control.
    
    Args:
        num_examples: Target number of examples
        min_score: Minimum quality score to accept
        max_attempts: Maximum generation attempts (default: 2x num_examples)
        
    Returns:
        List of high-quality training examples
    """
    if max_attempts is None:
        max_attempts = num_examples * 2
    
    generator = DataGenerator()
    validator = DatasetValidator()
    
    accepted_examples = []
    attempts = 0
    
    print(f"Generating {num_examples} high-quality examples (min score: {min_score})...")
    print(f"Maximum attempts: {max_attempts}")
    
    while len(accepted_examples) < num_examples and attempts < max_attempts:
        attempts += 1
        
        # Progress reporting
        if attempts % 100 == 0:
            acceptance_rate = len(accepted_examples) / max(1, attempts) * 100
            print(f"Progress: {len(accepted_examples)}/{num_examples} accepted, "
                  f"{attempts} attempts ({acceptance_rate:.1f}% acceptance rate)")
        
        try:
            # Generate example
            example = generator.generate_training_example()
            
            # Validate example
            validation_result = validator.validate_example(example)
            
            # Accept if meets quality threshold
            if validation_result["valid"] and validation_result["score"] >= min_score:
                accepted_examples.append(example)
                
                # Occasional detailed reporting
                if len(accepted_examples) % 250 == 0:
                    print(f"✅ Accepted example {len(accepted_examples)}: "
                          f"score={validation_result['score']:.3f}")
                    print(f"   Text: {example.text_description}")
            
        except Exception as e:
            print(f"Error generating example {attempts}: {e}")
            continue
    
    final_acceptance_rate = len(accepted_examples) / attempts * 100
    print(f"\nGeneration complete!")
    print(f"Accepted: {len(accepted_examples)}/{attempts} ({final_acceptance_rate:.1f}%)")
    
    return accepted_examples


def save_dataset_multiple_formats(examples: List[TrainingExample], base_path: str):
    """Save dataset in multiple formats for different use cases."""
    
    # Sanitize path to prevent traversal attacks
    base_path = os.path.normpath(base_path)
    if '..' in base_path or base_path.startswith('/'):
        raise ValueError("Invalid path: path traversal detected")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # 1. Full JSON format (complete data)
    full_data = {
        "metadata": {
            "num_examples": len(examples),
            "generated_at": time.time(),
            "generator_version": "1.0",
            "format": "full"
        },
        "examples": [example.to_dict() for example in examples]
    }
    
    full_path = f"{base_path}_full.json"
    with open(full_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved full dataset to {full_path}")
    
    # 2. Text-only format (for NLP models)
    text_data = {
        "metadata": {
            "num_examples": len(examples),
            "generated_at": time.time(),
            "format": "text_only"
        },
        "examples": [
            {
                "id": example.example_id,
                "text": example.text_description,
                "tags": example.scene.tags,
                "num_objects": len([obj for obj in example.scene.objects 
                                  if obj.object_type.value != "plane"])
            }
            for example in examples
        ]
    }
    
    text_path = f"{base_path}_text.json"
    with open(text_path, 'w') as f:
        json.dump(text_data, f, indent=2)
    print(f"Saved text-only dataset to {text_path}")
    
    # 3. Scene-only format (for physics validation)
    scene_data = {
        "metadata": {
            "num_examples": len(examples),
            "generated_at": time.time(),
            "format": "scenes_only"
        },
        "scenes": [example.scene.to_dict() for example in examples]
    }
    
    scene_path = f"{base_path}_scenes.json"
    with open(scene_path, 'w') as f:
        json.dump(scene_data, f, indent=2)
    print(f"Saved scenes-only dataset to {scene_path}")
    
    # 4. CSV format (for analysis)
    import csv
    csv_path = f"{base_path}_summary.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'example_id', 'text_description', 'num_objects', 'scenario_type',
            'complexity', 'tags', 'has_ramp', 'has_sphere', 'has_box'
        ])
        
        for example in examples:
            objects = [obj for obj in example.scene.objects if obj.object_type.value != "plane"]
            object_types = [obj.object_type.value for obj in objects]
            
            writer.writerow([
                example.example_id,
                example.text_description,
                len(objects),
                example.metadata.get('scenario_type', 'unknown'),
                example.metadata.get('complexity', 'unknown'),
                ','.join(example.scene.tags),
                'ramp' in object_types,
                'sphere' in object_types,
                'box' in object_types
            ])
    
    print(f"Saved CSV summary to {csv_path}")


def analyze_dataset(examples: List[TrainingExample]):
    """Analyze and report dataset statistics."""
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"Total examples: {len(examples)}")
    
    # Text length statistics
    text_lengths = [len(example.text_description) for example in examples]
    print(f"Text length - Min: {min(text_lengths)}, Max: {max(text_lengths)}, "
          f"Avg: {sum(text_lengths)/len(text_lengths):.1f}")
    
    # Scenario type distribution
    scenario_types = {}
    for example in examples:
        scenario_type = example.metadata.get('scenario_type', 'unknown')
        scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
    
    print("\nScenario type distribution:")
    for scenario_type, count in sorted(scenario_types.items()):
        percentage = count / len(examples) * 100
        print(f"  {scenario_type}: {count} ({percentage:.1f}%)")
    
    # Object count distribution
    object_counts = {}
    for example in examples:
        num_objects = len([obj for obj in example.scene.objects if obj.object_type.value != "plane"])
        object_counts[num_objects] = object_counts.get(num_objects, 0) + 1
    
    print("\nObject count distribution:")
    for count, freq in sorted(object_counts.items()):
        percentage = freq / len(examples) * 100
        print(f"  {count} objects: {freq} ({percentage:.1f}%)")
    
    # Tag analysis
    all_tags = []
    for example in examples:
        all_tags.extend(example.scene.tags)
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    print("\nMost common tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / len(examples) * 100
        print(f"  {tag}: {count} ({percentage:.1f}%)")
    
    # Complexity distribution
    complexity_counts = {}
    for example in examples:
        complexity = example.metadata.get('complexity', 'unknown')
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    print("\nComplexity distribution:")
    for complexity, count in sorted(complexity_counts.items()):
        percentage = count / len(examples) * 100
        print(f"  {complexity}: {count} ({percentage:.1f}%)")


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser(description='Generate physics training dataset')
    parser.add_argument('--num_examples', type=int, default=1000,
                       help='Number of examples to generate (default: 1000)')
    parser.add_argument('--min_score', type=float, default=0.5,
                       help='Minimum quality score (default: 0.5)')
    parser.add_argument('--output_path', type=str, default='data/physics_dataset',
                       help='Output path prefix (default: data/physics_dataset)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick generation with lower quality threshold')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick generation
    if args.quick:
        args.min_score = 0.3
        print("Quick mode: Using lower quality threshold")
    
    print(f"Starting dataset generation...")
    print(f"Target examples: {args.num_examples}")
    print(f"Minimum score: {args.min_score}")
    print(f"Output path: {args.output_path}")
    
    start_time = time.time()
    
    # Generate dataset
    examples = generate_filtered_dataset(
        num_examples=args.num_examples,
        min_score=args.min_score
    )
    
    if len(examples) == 0:
        print("❌ No examples generated! Check generation parameters.")
        return
    
    # Validate final dataset
    print("\nValidating final dataset...")
    validator = DatasetValidator()
    validation_result = validator.validate_dataset(examples)
    
    print(f"Final validation - Valid: {validation_result['dataset_valid']}")
    print(f"Validity rate: {validation_result['validity_rate']:.1%}")
    print(f"Average score: {validation_result['average_score']:.3f}")
    
    # Save in multiple formats
    print("\nSaving dataset...")
    save_dataset_multiple_formats(examples, args.output_path)
    
    # Analyze dataset
    analyze_dataset(examples)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Dataset generation complete!")
    print(f"Generated {len(examples)} examples in {elapsed_time:.1f} seconds")
    print(f"Average time per example: {elapsed_time/len(examples):.2f} seconds")


if __name__ == "__main__":
    main()
