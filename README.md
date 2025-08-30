# Neophysics - Natural Language Physics Engine

A machine learning-powered 3D physics simulation engine that interprets natural language commands and generates realistic physics scenarios. Built with PyBullet for physics simulation, a finetuned T5 model or OpenAI's gpt-oss-20b for advanced natural language understanding with chain-of-thought reasoning.

### Important - OpenAI gpt-oss-20b is not fine-tuned, more data is needed for T5 model to be generalizable

Due to a lack of compute power and hardware limitations, I was unable to fine-tune gpt-oss-20b for the purposes of this project. Please feel free to fine-tune with the provided scripts and utilize gpt-oss-20b as you wish. If you wish to use this program as is, please run `python src/main.py --model t5-trained` to use the fine tuned T5 model. 

Also, the fine-tuned T5 model is trained on very narrow use cases. As such, the current trained model will not be available. The trained model will be published in the upcoming weeks after further training is completed. In the meantime, feel free to train the model on your own. 

I realize these things make this project completely dysfunctional for most people. However, for the sake of documentation and building in public, I still chose to make this public.

## Overview

Neophysics translates plain English descriptions into structured physics simulations. Users can create physics scenarios by typing commands like "create a ball on a ramp" and watch the simulation unfold in real-time.

**Things it can do:**
- Natural language to physics translation using OpenAI gpt-oss-20b with chain-of-thought reasoning
- Real-time 3D physics simulation with PyBullet
- Interactive GUI for command input and simulation control
- Local inference with efficient MoE architecture (21B parameters, ~3.6B active)
- Advanced reasoning capabilities for complex physics scenarios

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed (https://ollama.ai/download)
- 16GB+ RAM recommended (for gpt-oss-20b)
- GPU recommended (CUDA-compatible for optimal performance)
- Fallback to T5-small available for lower-spec systems

### Installation

#### Quick Setup (Recommended)
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

# Install Ollama (if not already installed)
# Download from: https://ollama.ai/download
# Then pull the model:
ollama pull gpt-oss:20b
```

#### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup Ollama model
ollama pull gpt-oss:20b
```

### Running the Application

```bash
# Launch with auto-detection (default)
python src/main.py

# Force specific model
python src/main.py --model gpt-oss
python src/main.py --model t5-trained

# Test physics engine directly
python src/physics_engine.py
```

### Model Backends

Neophysics now supports multiple model backends:

- **gpt-oss-20b** (Recommended): OpenAI's open-source model via Ollama
- **t5-small**: Lightweight fallback for resource-constrained systems

Requires Ollama installation: `ollama pull gpt-oss:20b`

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
| `nlp_model.py` | Language Model | OpenAI gpt-oss-20b wrapper with T5 fallback |
| `ml_physics_bridge.py` | ML-Physics Integration | Action sequence parsing, object mapping |
| `realtime_simulator.py` | Simulation Engine | Real-time physics, data capture |
| `physics_validator.py` | Validation System | Physics plausibility checking |

### Data Flow

```
Natural Language → GPT-OSS-20B → Chain-of-Thought → Action Sequence → Physics Objects → 3D Simulation
```

**Example Pipeline:**
```
Input: "create a ball on a ramp"
↓
GPT-OSS-20B Reasoning: "I need to create a sphere and a ramp, position the sphere above the ramp..."
↓
Action Output: "CREATE id=obj1 type=sphere pos=(0,0,1) ...; CREATE id=obj2 type=ramp ..."
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
- **Transformers**: Hugging Face model implementations
- **vLLM**: Efficient inference for gpt-oss-20b (optional)
- **PyBullet**: 3D physics simulation
- **NumPy**: Numerical computations
- **Tkinter**: GUI framework (built-in)
- **tqdm**: Progress tracking

### Performance
- **Model Size**: ~21B parameters (~40GB), ~3.6B active per token
- **Inference Time**: ~200ms per command (with reasoning)
- **Memory Usage**: ~16GB for model + ~500MB simulation
- **Physics FPS**: 60-240 Hz (configurable)
- **Reasoning**: Chain-of-thought with adjustable levels (low/medium/high)
- **Fallback**: T5-small (~60MB) for resource-constrained systems

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
