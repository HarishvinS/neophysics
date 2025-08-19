# Learnable Physics Engine

A local 3D physics sandbox where users can create and simulate physics scenarios using natural language commands processed by learned neural networks.

## Project Overview

**Goal**: Build a local application that understands natural language physics commands and creates realistic 3D simulations.

**Example Use Case**: 
- Input: "create a u-shaped ramp and place a 2kg ball at the top right with 2 m/s velocity"
- Output: 3D simulation showing realistic physics behavior

## Project Structure

```
neophysics/
├── src/                    # Source code
├── data/                   # Training data and datasets
├── models/                 # Saved ML models
├── assets/                 # 3D models, textures, etc.
├── examples/               # Example scripts and demos
├── venv/                   # Virtual environment
├── physics_engine_plan.md  # Detailed project plan
└── README.md              # This file
```

## Tech Stack

- **Core**: Python 3.8+
- **GUI**: Tkinter (built-in)
- **3D Graphics**: PyBullet (physics simulation)
- **ML**: PyTorch (local inference)
- **Math**: NumPy, SciPy

## Development Phases

### Phase 1: Foundation & Basic Physics (Weeks 1-3)
- [x] Development environment setup
- [x] Basic physics simulation
- [ ] Data generation pipeline
- [ ] Simple ML model

### Phase 2: Core Learning System (Weeks 4-6)
- [ ] ML-Physics integration
- [ ] Quality assessment system
- [ ] Continuous learning

### Phase 3: Advanced Features (Weeks 7-9)
- [ ] Enhanced object types
- [ ] Natural language processing
- [ ] Professional UI

## Getting Started

1. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the basic application:
   ```bash
   python src/main.py
   ```

4. Try the demos:
   ```bash
   # Week 1: Basic Physics
   python examples/demo_week1.py

   # Week 2: Data Generation
   python examples/demo_week2.py

   # Week 3: ML Pipeline
   python examples/demo_week3.py

   # Week 4: Complete Integration
   python examples/demo_week4.py

   # Week 5: Advanced Understanding
   python examples/demo_week5.py

   # Week 6: Continuous Learning
   python examples/demo_week6.py

   # Week 7: Advanced Physics Understanding
   python examples/demo_week7.py
   ```

5. Generate training data:
   ```bash
   # Quick test dataset (100 examples)
   python src/generate_dataset.py --num_examples 100 --quick

   # Full training dataset (1000+ examples)
   python src/generate_dataset.py --num_examples 1000
   ```

6. Train the ML model:
   ```bash
   # Quick training (5 epochs)
   python src/train_model.py --quick

   # Full training (20 epochs)
   python src/train_model.py --epochs 20
   ```

7. Use the interactive interface:
   ```bash
   # Launch GUI for live text-to-physics
   python src/interactive_interface.py
   ```

8. Run comprehensive tests:
   ```bash
   # End-to-end pipeline test
   python examples/test_end_to_end.py
   ```

9. Run tests to verify everything works:
   ```bash
   python examples/test_basic_functionality.py
   ```

## Current Status

### Week 1: COMPLETED ✅
✅ Virtual environment created and dependencies installed
✅ Basic project structure established
✅ PyBullet physics engine integration
✅ 3D object creation (spheres, boxes, ramps)
✅ Tkinter GUI with manual controls
✅ Comprehensive testing suite
✅ Real-time physics simulation working

### Week 2: COMPLETED ✅
✅ Comprehensive scene representation system
✅ Natural language text generation with templates
✅ Automated physics scenario generation
✅ Quality validation and filtering
✅ Large-scale dataset generation (1000+ examples)
✅ Multiple output formats (JSON, CSV)

### Week 3: COMPLETED ✅
✅ Neural network architecture (67M parameters)
✅ Text encoder using pre-trained transformers
✅ Scene decoder with multi-task outputs
✅ Complete training pipeline with loss functions
✅ Comprehensive evaluation system
✅ Physics plausibility assessment
✅ End-to-end text-to-scene translation

### Week 4: COMPLETED ✅
✅ ML-Physics bridge with automatic object creation
✅ Real-time simulation with data capture
✅ Physics validation and plausibility scoring
✅ Interactive GUI for live text-to-physics
✅ End-to-end pipeline validation
✅ Performance optimization (24+ commands/second)

### Week 5: COMPLETED ✅
✅ Dynamic scene representation (no fixed-size limits)
✅ Physics reasoning engine (causal understanding)
✅ Advanced relational understanding (beyond templates)
✅ Comprehensive generalization testing
✅ Architectural limitation analysis and solutions

### Week 6: COMPLETED ✅
✅ User feedback collection and analysis system
✅ Active learning with uncertainty and diversity sampling
✅ Comprehensive performance tracking and monitoring
✅ Automated self-improvement loop
✅ Continuous learning validation framework

### Week 7: COMPLETED ✅
✅ Advanced mechanical objects with constraints and compound structures
✅ Realistic material properties with comprehensive physics databases
✅ Multi-step physics reasoning and chain reaction prediction
✅ Deep causal understanding of fundamental physics laws
✅ Complex scenario analysis and sophisticated interaction modeling

### Next: Week 8 - Natural Language Enhancement
⏳ Better command understanding and context awareness
⏳ Multi-step command handling
⏳ Natural conversation about physics

## License

MIT License - See LICENSE file for details
