# Neophysics - Learnable Physics Engine

A 3D physics simulation engine that understands natural language commands using machine learning. Create physics scenarios by typing commands like "create a ball on a ramp" and watch realistic simulations unfold.

## Features

- **Natural Language Interface**: Type physics commands in plain English
- **Real-time 3D Simulation**: PyBullet-powered physics with visual feedback
- **Machine Learning**: T5 transformer model translates text to physics actions
- **Interactive GUI**: Tkinter-based interface with live simulation controls
- **Extensible Architecture**: Modular design for adding new objects and behaviors

## Quick Start

### Prerequisites
- Python 3.8+
- Windows/Linux/macOS

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neophysics.git
cd neophysics
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

**Interactive GUI Mode:**
```bash
python src/main.py
```

**Basic Physics Testing:**
```bash
python src/physics_engine.py
```

## Usage Examples

Once the GUI is running, try these commands:

- `create a ball` - Creates a red sphere
- `add a sphere on a ramp` - Places a ball on an inclined surface
- `place two boxes` - Creates two box objects
- `create a bouncy ball and a wooden ramp` - Multiple objects with materials
- `add three metal spheres` - Multiple objects with specific materials

## Training Your Own Model

### 1. Generate Training Data
```bash
# Quick dataset (100 examples)
python src/generate_dataset.py --num_examples 100

# Full dataset (1000+ examples)
python src/generate_dataset.py --num_examples 1000
```

### 2. Train the Model
```bash
# Quick training (3 epochs)
python src/train_model.py --epochs 3

# Full training (20 epochs)
python src/train_model.py --epochs 20 --batch_size 8
```

### 3. Use Trained Model
The trained model is automatically loaded by `main.py` if available in `models/physics_model/`.

## Project Structure

```
neophysics/
├── src/                           # Source code
│   ├── main.py                   # Main GUI application
│   ├── physics_engine.py         # PyBullet physics simulation
│   ├── nlp_model.py              # T5 transformer model
│   ├── generate_dataset.py       # Training data generation
│   ├── train_model.py            # Model training script
│   ├── ml_physics_bridge.py      # ML-Physics integration
│   ├── realtime_simulator.py     # Real-time simulation engine
│   └── dynamic_scene_representation.py  # Scene data structures
├── models/                        # Trained ML models
├── data/                         # Generated datasets
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Architecture

### Core Components

1. **Physics Engine** (`physics_engine.py`)
   - PyBullet-based 3D physics simulation
   - Supports spheres, boxes, ramps with realistic materials
   - Gravity, collisions, friction, and restitution

2. **NLP Model** (`nlp_model.py`)
   - T5 transformer for text-to-action translation
   - Converts natural language to structured action sequences
   - Pre-trained on physics command datasets

3. **ML-Physics Bridge** (`ml_physics_bridge.py`)
   - Translates ML predictions to physics objects
   - Handles object creation, positioning, and relationships
   - Validates physics plausibility

4. **Interactive Interface** (`main.py`)
   - Tkinter GUI with command input and status display
   - Real-time simulation controls and feedback
   - Example commands and validation tools

### Data Flow

```
Text Input → T5 Model → Action Sequence → Physics Objects → 3D Simulation
```

Example:
```
"create a ball on a ramp" 
→ "CREATE id=obj1 type=sphere pos=(0,0,1) ...; CREATE id=obj2 type=ramp ...; RELATE subject_id=obj1 type=on target_id=obj2"
→ Physics objects with proper positioning
→ Visual 3D simulation
```

## Technical Details

### Supported Objects
- **Spheres**: Configurable radius, mass, material properties
- **Boxes**: Configurable dimensions, mass, material properties  
- **Ramps**: Inclined planes with adjustable angle and size

### Materials
- **Wood**: Medium friction, low restitution
- **Metal**: High mass, medium friction
- **Rubber**: High restitution (bouncy)

### Relationships
- **On**: Places one object on top of another
- **Next to**: Places objects adjacent to each other

## Development

### Adding New Object Types

1. Extend `ObjectType` enum in `dynamic_scene_representation.py`
2. Add creation logic in `physics_engine.py`
3. Update ML model training data in `generate_dataset.py`

### Adding New Materials

1. Extend `MaterialType` enum
2. Add material properties in physics engine
3. Update data generation templates

### Extending NLP Understanding

1. Add new command patterns to `generate_dataset.py`
2. Retrain the model with expanded dataset
3. Update action sequence parsing logic

## Dependencies

- **torch**: PyTorch for ML model training and inference
- **transformers**: Hugging Face transformers (T5 model)
- **pybullet**: 3D physics simulation engine
- **tkinter**: GUI framework (built into Python)
- **numpy**: Numerical computations
- **tqdm**: Progress bars for training

## Performance

- **Model Size**: ~60MB (T5-small)
- **Inference Speed**: ~24 commands/second
- **Training Time**: ~10 minutes for 1000 examples (CPU)
- **Memory Usage**: ~500MB during simulation

## Limitations

- Limited to basic geometric shapes
- Simple spatial relationships
- No complex multi-step physics reasoning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.