# Neophysics - Natural Language Physics Engine

A machine learning-powered 3D physics simulation engine that interprets natural language commands and generates realistic physics scenarios. Built with PyBullet for physics simulation and T5 transformer for natural language understanding.

## Overview

Neophysics translates plain English descriptions into structured physics simulations. Users can create physics scenarios by typing commands like "create a ball on a ramp" and watch the simulation unfold in real-time.

**Things it can do:**
- Natural language to physics translation using T5 transformer
- Real-time 3D physics simulation with PyBullet
- Interactive GUI for command input and simulation control
- Extensible training pipeline for custom physics scenarios

## Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (for faster training)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neophysics.git
cd neophysics

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch interactive GUI
python src/main.py

# Test physics engine directly
python src/physics_engine.py
```

## Usage

### Basic Commands

The GUI accepts natural language commands for creating physics scenarios:

Some examples: 

```
create a ball                   # Single red sphere
add a sphere on a ramp          # Ball placed on inclined surface
place two boxes                 # Two box objects
create a bouncy ball            # Sphere with high restitution
add three metal spheres         # Multiple objects with material properties
make a ball and cube collide    # Objects with initial velocities
```

### Interactive Mode

The main interface provides:
- **Command Input**: Text field for natural language commands
- **Simulation Controls**: Play, pause, clear scene
- **Status Panel**: Real-time feedback and error messages
- **Example Commands**: Quick-start buttons for common scenarios

## Training Pipeline

### 1. Data Generation

```bash
# Generate synthetic training data
python src/generate_dataset.py --num_examples 1000

# Interactive scene creation
python src/generate_dataset.py --interactive

# Augment existing data
python src/generate_dataset.py --augment data/training_data.json --multiplier 10
```

### 2. Model Training

```bash
# Train T5 model (basic)
python src/train_model.py --epochs 10 --batch_size 4

# Advanced training with custom parameters
python src/train_model.py \
  --data_path data/training_data.json \
  --epochs 20 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --test_split_size 0.2
```

### 3. Model Usage

Trained models are automatically loaded from `models/physics_model/`. If no trained model exists, the system falls back to the base T5-small model.

## Architecture

### Core Modules

| Module | Purpose | Key Features |
|--------|---------|-------------|
| `main.py` | GUI Application | Tkinter interface, command processing |
| `physics_engine.py` | Physics Simulation | PyBullet integration, object creation |
| `nlp_model.py` | Language Model | T5 transformer wrapper |
| `ml_physics_bridge.py` | ML-Physics Integration | Action sequence parsing, object mapping |
| `realtime_simulator.py` | Simulation Engine | Real-time physics, data capture |
| `physics_validator.py` | Validation System | Physics plausibility checking |

### Data Flow

```
Natural Language → T5 Model → Action Sequence → Physics Objects → 3D Simulation
```

**Example Pipeline:**
```
Input: "create a ball on a ramp"
↓
T5 Model Output: "CREATE id=obj1 type=sphere pos=(0,0,1) ...; CREATE id=obj2 type=ramp ..."
↓
Parsed Objects: [Sphere(pos=(0,0,1)), Ramp(pos=(0,0,0), angle=0.3)]
↓
PyBullet Simulation: Visual 3D physics with gravity and collisions
```

### Action Sequence Format

The system uses a structured action language:

```
CREATE id=obj1 type=sphere pos=(0.0,0.0,1.0) rot=(0.0,0.0,0.0) scale=(0.2,0.2,0.2) mass=1.0 material=wood;
CREATE id=obj2 type=ramp pos=(0.0,0.0,0.0) rot=(0.0,0.3,0.0) scale=(2.0,0.2,1.0) mass=0.0 material=wood;
RELATE subject_id=obj1 type=on target_id=obj2;
```

## Supported Features

### Object Types
- **Spheres**: Configurable radius, mass, material
- **Boxes**: Configurable dimensions, mass, material
- **Ramps**: Inclined planes with adjustable angle

### Materials
- **Wood**: Balanced friction and restitution
- **Metal**: High mass, medium friction
- **Rubber**: High restitution (bouncy)
- **Ice**: Low friction
- **Plastic**: Medium properties
- **Glass**: Low restitution
- **Stone**: High friction, low restitution

### Spatial Relationships
- **On**: Object placement on top of another
- **Next to**: Adjacent object positioning

## Development

### Project Structure

```
neophysics/
├── src/
│   ├── main.py                          # Main GUI application
│   ├── physics_engine.py                # PyBullet physics engine
│   ├── nlp_model.py                     # T5 model wrapper
│   ├── ml_physics_bridge.py             # ML-physics integration
│   ├── realtime_simulator.py            # Real-time simulation
│   ├── physics_validator.py             # Validation system
│   ├── dynamic_scene_representation.py  # Data structures
│   ├── generate_dataset.py              # Data generation
│   └── train_model.py                   # Model training
├── models/                              # Trained models
├── data/                                # Training datasets
└── requirements.txt                     # Dependencies
```

### Extending the System

**Add New Object Types:**
1. Extend `ObjectType` enum in `dynamic_scene_representation.py`
2. Add creation logic in `physics_engine.py`
3. Update training data generation in `generate_dataset.py`

**Add New Materials:**
1. Extend `MaterialType` enum
2. Add material properties in `ml_physics_bridge.py`
3. Update data generation templates

**Improve Language Understanding:**
1. Generate more diverse training data
2. Add new command patterns to `generate_dataset.py`
3. Retrain the T5 model with expanded dataset

## Technical Specifications

### Dependencies
- **PyTorch**: ML model training and inference
- **Transformers**: Hugging Face T5 implementation
- **PyBullet**: 3D physics simulation
- **NumPy**: Numerical computations
- **Tkinter**: GUI framework (built-in)
- **tqdm**: Progress tracking

### Performance
- **Model Size**: ~60MB (T5-small base)
- **Inference Time**: ~40ms per command
- **Training Time**: ~10 minutes for 1000 examples (CPU)
- **Memory Usage**: ~500MB during simulation
- **Physics FPS**: 60-240 Hz (configurable)

### Limitations
- Basic geometric shapes only (sphere, box, ramp)
- Simple spatial relationships
- Single-step command processing
- No persistent scene memory

## Troubleshooting

**Model not loading:**
- Ensure `models/physics_model/` contains trained model files
- Check PyTorch and Transformers versions
- Verify sufficient memory (4GB+ recommended)

**Physics simulation issues:**
- Update PyBullet to latest version
- Check OpenGL drivers for GUI mode
- Reduce simulation complexity for better performance

**Training problems:**
- Verify training data format in `data/training_data.json`
- Ensure sufficient disk space for model checkpoints
- Use smaller batch size if running out of memory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and add tests
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## License

MIT License - see LICENSE file for details.