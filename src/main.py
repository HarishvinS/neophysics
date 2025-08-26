"""
Interactive Interface for Text-to-Physics System
Combines text input, ML prediction, and live physics simulation in a unified GUI.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
from typing import Optional, Dict, List, Any
import re

from ml_physics_bridge import MLPhysicsBridge
from realtime_simulator import RealTimeSimulator
from physics_validator import PhysicsValidator
from nlp_model import Seq2SeqModel, T5Seq2SeqModel  # <-- Use OpenAI gpt-oss-20b model
from model_config import ModelConfig
from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject, ObjectType, MaterialType, Vector3


class InteractivePhysicsApp:
    """Main interactive application for text-to-physics system."""
    
    def __init__(self, model_override=None):
        """Initialize the interactive application."""
        self.root = tk.Tk()
        self.root.title("Learnable Physics Engine - Interactive Mode")
        self.root.geometry("1000x700")
        self.model_override = model_override
        
        # Components
        self.model = None
        self.bridge = None
        self.simulator = None
        self.validator = None

        # State
        self.model_loaded = False
        self.physics_initialized = False
        self.simulation_running = False
        
        # Setup UI
        self.setup_ui()
        
        # Try to load model automatically
        self._load_model_async()
        
        print("Neophysics Interactive Engine initialized!")
        print("Tip: Run 'python src/model_config.py' to configure your model backend")
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Header
        self.setup_header(main_frame)
        
        # Control panel (left side)
        self.setup_control_panel(main_frame)
        
        # Status and results (right side)
        self.setup_status_panel(main_frame)
    
    def setup_header(self, parent):
        """Setup header section."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Neophysics", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Status indicators
        status_frame = ttk.Frame(header_frame)
        status_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.model_status = ttk.Label(status_frame, text="Model: Not Loaded", font=('Arial', 9))
        self.model_status.grid(row=0, column=0, padx=(0, 10))
        
        self.physics_status = ttk.Label(status_frame, text="Physics: Not Ready", font=('Arial', 9))
        self.physics_status.grid(row=0, column=1)
    
    def setup_control_panel(self, parent):
        """Setup control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # Text input section
        ttk.Label(control_frame, text="Enter physics command:", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(
            control_frame, 
            textvariable=self.command_var, 
            font=('Arial', 11),
            width=40
        )
        self.command_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.command_entry.bind('<Return>', lambda e: self.execute_command())
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        self.execute_btn = ttk.Button(
            button_frame, 
            text="Execute", 
            command=self.execute_command,
            style='Accent.TButton'
        )
        self.execute_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.validate_btn = ttk.Button(
            button_frame, 
            text="Validate", 
            command=self.validate_command
        )
        self.validate_btn.grid(row=0, column=1, padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_scene
        )
        self.clear_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Simulation controls
        sim_frame = ttk.LabelFrame(control_frame, text="Simulation", padding="5")
        sim_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        sim_frame.columnconfigure(1, weight=1)
        
        ttk.Label(sim_frame, text="Duration:").grid(row=0, column=0, sticky=tk.W)
        
        self.duration_var = tk.StringVar(value="3.0")
        duration_entry = ttk.Entry(sim_frame, textvariable=self.duration_var, width=8)
        duration_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 10))
        
        ttk.Label(sim_frame, text="seconds").grid(row=0, column=2, sticky=tk.W)
        
        # Example commands
        examples_frame = ttk.LabelFrame(control_frame, text="Example Commands", padding="5")
        examples_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        examples = [
            "create a ball",
            "add a sphere on a ramp",
            "place two boxes",
            "create a bouncy ball and a wooden ramp",
            "add three metal spheres"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(
                examples_frame, 
                text=example, 
                command=lambda cmd=example: self.set_command(cmd),
                width=35
            )
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=1)
    
    def setup_status_panel(self, parent):
        """Setup status and results panel."""
        status_frame = ttk.LabelFrame(parent, text="Status & Results", padding="10")
        status_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # Status log
        self.status_log = scrolledtext.ScrolledText(
            status_frame, 
            height=25, 
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.status_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var, 
            mode='determinate'
        )
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initial messages
        self.log_message("Welcome to the Interactive Physics Engine!")
        self.log_message("Loading ML model...")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add a message to the status log."""
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding
        if level == "ERROR":
            prefix = "[ERROR]"
        elif level == "SUCCESS":
            prefix = "[SUCCESS]"
        elif level == "WARNING":
            prefix = "[WARNING]"
        else:
            prefix = "[INFO]"
        
        log_entry = f"[{timestamp}] {prefix} {message}\n"
        
        self.status_log.insert(tk.END, log_entry)
        self.status_log.see(tk.END)
        self.root.update()
    
    def _update_status_indicators(self):
        """Update status indicators in header."""
        if self.model_loaded:
            self.model_status.config(text="Model: Loaded")
        else:
            self.model_status.config(text="Model: Not Loaded")
        
        if self.physics_initialized:
            self.physics_status.config(text="Physics: Ready")
        else:
            self.physics_status.config(text="Physics: Not Ready")
    
    def _load_model_async(self):
        """Load model in background thread."""
        def load_model():
            try:
                self.progress_var.set(20)
                
                # Load model based on override or configuration
                if self.model_override and self.model_override != 'auto':
                    backend = self.model_override
                    self.log_message(f"Using specified backend: {backend}")
                else:
                    config_manager = ModelConfig()
                    model_config = config_manager.load_config()
                    backend = model_config.get('selected_backend', 'auto-detect')
                    
                    if backend == 'auto-detect':
                        capabilities = config_manager.detect_system_capabilities()
                        backend = capabilities['recommended_backend']
                        self.log_message(f"Auto-detected backend: {backend}")
                
                try:
                    if backend == 'gpt-oss':
                        self.log_message("Loading GPT-OSS-20B model...")
                        self.model = Seq2SeqModel(model_name="gpt-oss:20b")
                        self.log_message("GPT-OSS-20B model loaded successfully!", "SUCCESS")
                    elif backend == 'gpt-oss-lora':
                        self.log_message("Loading GPT-OSS-20B LoRA model...")
                        from nlp_model import GPTOSSLoRAModel
                        self.model = GPTOSSLoRAModel()
                        self.log_message("GPT-OSS-20B LoRA model loaded successfully!", "SUCCESS")
                    elif backend == 't5-trained':
                        self.log_message("Loading trained T5 model...")
                        model_path = "models/physics_model"
                        if os.path.exists(model_path):
                            self.model = T5Seq2SeqModel.load(model_path)
                            self.log_message("Trained T5 model loaded", "SUCCESS")
                        else:
                            self.log_message("No trained model found, using T5-small", "WARNING")
                            self.model = T5Seq2SeqModel(model_name="t5-small")
                    else:  # t5-small or fallback
                        self.log_message("Loading T5-small model...")
                        self.model = T5Seq2SeqModel(model_name="t5-small")
                        self.log_message("T5-small model loaded", "SUCCESS")
                            
                except Exception as e:
                    self.log_message(f"Failed to load {backend}: {str(e)}", "WARNING")
                    self.log_message("Falling back to T5-small...", "WARNING")
                    self.model = T5Seq2SeqModel(model_name="t5-small")
                    self.log_message("T5-small loaded as fallback", "SUCCESS")
                
                self.progress_var.set(60)
                
                # Initialize components
                self.bridge = MLPhysicsBridge(self.model, use_gui=True) # Create the bridge
                self.bridge.initialize_physics() # Immediately initialize the physics engine
                self.physics_initialized = True
                
                # Now that the physics engine is ready, we can safely create the simulator and validator
                self.simulator = RealTimeSimulator(self.bridge, fps=60)
                self.validator = PhysicsValidator(self.bridge, self.simulator)
                self.progress_var.set(80)
                
                # Add simulator callbacks
                self.simulator.add_event_callback(self.on_simulation_event)
                
                self.progress_var.set(100)
                
                self.model_loaded = True
                self.log_message("System ready! You can now enter physics commands.", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"Failed to load model: {str(e)}", "ERROR")
            
            finally:
                self._update_status_indicators()
                self.progress_var.set(0)
        
        # Start loading in background
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def set_command(self, command: str):
        """Set command in text field."""
        self.command_var.set(command)
    
    def execute_command(self):
        """Execute the current command."""
        command = self.command_var.get().strip()
        if not command:
            return
        
        if not self.model_loaded:
            self.log_message("Please wait for model to load", "WARNING")
            return
        
        self.log_message(f"Processing: '{command}'")
        
        def execute():
            try:
                # Initialize physics if needed
                if not self.physics_initialized:
                    self.bridge.initialize_physics()
                    self.physics_initialized = True
                    self._update_status_indicators()

                # 1. Get action sequence from the new model
                self.log_message("1. Generating action sequence from command...")
                action_sequence_str = self.model.generate(command)
                self.log_message(f"   > Predicted sequence: {action_sequence_str}")
                
                # Check if model returned natural language instead of action sequence
                if not any(keyword in action_sequence_str.upper() for keyword in ['CREATE', 'RELATE', 'ID=', 'TYPE=']):
                    self.log_message("Model returned natural language instead of action sequence. Model may need more training.", "WARNING")

                # 2. Build the scene from this action sequence
                self.log_message("2. Building scene from action sequence...")
                scene = self._build_scene_from_action_sequence(action_sequence_str)

                if not scene.get_object_count():
                    self.log_message("No objects were created from the command.", "WARNING")
                    return

                self.log_message(f"3. Rendering {scene.get_object_count()} objects...")
                self.bridge.scene_to_physics(scene)

                duration = float(self.duration_var.get())
                self.log_message(f"4. Running simulation for {duration}s...")
                self.start_simulation(duration)
                
            except Exception as e:
                self.log_message(f"Execution error: {str(e)}", "ERROR")
                import traceback
                traceback.print_exc()
        
        # Run in background thread
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
    
    def validate_command(self):
        """Validate the current command."""
        command = self.command_var.get().strip()
        if not command:
            return
        
        if not self.model_loaded:
            self.log_message("Please wait for model to load", "WARNING")
            return
        
        self.log_message(f"Validating: '{command}'")
        
        def validate():
            try:
                duration = float(self.duration_var.get())
                result = self.validator.validate_prediction(command, duration)
                
                # Log results
                self.log_message(f"Validation Score: {result.validation_score:.3f}")
                self.log_message(f"  Prediction Valid: {result.prediction_valid}")
                self.log_message(f"  Physics Plausible: {result.physics_plausible}")
                self.log_message(f"  Simulation Successful: {result.simulation_successful}")
                
                if result.details.get('errors'):
                    for error in result.details['errors'][:3]:  # Show first 3 errors
                        self.log_message(f"  Error: {error}", "WARNING")
                
            except Exception as e:
                self.log_message(f"Validation error: {str(e)}", "ERROR")
        
        # Run in background thread
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _parse_action_sequence(self, seq_str: str) -> List[Dict[str, Any]]:
        """Parses the action sequence string from the model into a list of action dicts."""
        actions = []
        
        # Handle case where model returns natural language instead of action sequence
        if not any(keyword in seq_str.upper() for keyword in ['CREATE', 'RELATE', 'ID=', 'TYPE=']):
            self.log_message(f"Model returned natural language instead of action sequence: {seq_str}", "WARNING")
            # Try to create a simple default object based on common words
            if any(word in seq_str.lower() for word in ['ball', 'sphere']):
                return [{'type': 'CREATE', 'params': {'id': 'obj1', 'type': 'sphere', 'pos': '(0,0,1)', 'rot': '(0,0,0)', 'scale': '(0.2,0.2,0.2)', 'mass': '1.0', 'material': 'wood'}}]
            elif any(word in seq_str.lower() for word in ['box', 'cube']):
                return [{'type': 'CREATE', 'params': {'id': 'obj1', 'type': 'box', 'pos': '(0,0,1)', 'rot': '(0,0,0)', 'scale': '(0.2,0.2,0.2)', 'mass': '1.0', 'material': 'wood'}}]
            else:
                return [{'type': 'CREATE', 'params': {'id': 'obj1', 'type': 'sphere', 'pos': '(0,0,1)', 'rot': '(0,0,0)', 'scale': '(0.2,0.2,0.2)', 'mass': '1.0', 'material': 'wood'}}]
        
        # Split by semicolon to get individual actions
        action_strs = [s.strip() for s in seq_str.split(';') if s.strip()]
        
        for action_str in action_strs:
            # Split by first space to get action type and parameters
            parts = action_str.split(' ', 1)
            if len(parts) < 2:
                continue
                
            action_type = parts[0].upper()
            params_str = parts[1]
            
            params = {}
            # Use regex to find key=value pairs, handling tuples properly
            param_matches = re.finditer(r'(\w+)=(\([^)]*\)|[^\s]+)', params_str)
            for p_match in param_matches:
                key = p_match.group(1)
                value = p_match.group(2).strip()
                params[key] = value
            
            if action_type and params:
                actions.append({'type': action_type, 'params': params})
        return actions

    def _build_scene_from_action_sequence(self, action_sequence_str: str) -> DynamicPhysicsScene:
        """Builds a DynamicPhysicsScene from a predicted action sequence string."""
        scene = DynamicPhysicsScene("predicted_scene")
        actions = self._parse_action_sequence(action_sequence_str)
        
        for action in actions:
            if action['type'] == 'CREATE':
                params = action['params']
                try:
                    # Safely parse tuple values
                    pos_tuple = eval(params.get('pos', '(0,0,1)'))
                    rot_tuple = eval(params.get('rot', '(0,0,0)'))
                    scale_tuple = eval(params.get('scale', '(0.5,0.5,0.5)'))
                    
                    # Extract material name (remove color if present)
                    material_str = params.get('material', 'wood')
                    if ' ' in material_str:
                        material_str = material_str.split()[0]

                    obj = DynamicPhysicsObject(
                        object_id=params.get('id', f"obj_{len(scene.objects)}"),
                        object_type=ObjectType(params.get('type', 'box')),
                        position=Vector3(*pos_tuple),
                        rotation=Vector3(*rot_tuple),
                        scale=Vector3(*scale_tuple),
                        mass=float(params.get('mass', 1.0)),
                        material=MaterialType(material_str)
                    )
                    scene.add_object(obj)
                    self.log_message(f"     > Created '{obj.object_id}' ({obj.object_type.value})")
                except Exception as e:
                    self.log_message(f"     > Failed to create object from params {params}: {e}", "WARNING")
            
            elif action['type'] == 'RELATE':
                # Placeholder for future relationship handling
                params = action['params']
                self.log_message(f"     > Relationship (not yet implemented): {params.get('subject_id')} "
                                 f"{params.get('type')} {params.get('target_id')}")

        return scene

    def start_simulation(self, duration: float):
        """Helper to start the simulation."""
        self.simulation_running = True
        self.simulator.start_simulation(duration=duration, record=True)
    
    def clear_scene(self):
        """Clear the physics scene."""
        if self.bridge:
            self.bridge.clear_scene()
            self.log_message("Scene cleared")
    
    def on_simulation_event(self, event_type: str, data: dict):
        """Handle simulation events."""
        if event_type == "simulation_stopped":
            self.simulation_running = False
            elapsed = data.get('elapsed_time', 0)
            steps = data.get('total_steps', 0)
            self.log_message(f"Simulation completed: {elapsed:.2f}s, {steps} steps", "SUCCESS")
            
            # Analyze motion
            if hasattr(self, 'simulator') and self.simulator:
                analysis = self.simulator.analyze_motion()
                if 'objects' in analysis:
                    objects_moved = sum(1 for obj in analysis['objects'].values() 
                                      if obj.get('total_displacement', 0) > 0.1)
                    self.log_message(f"Analysis: {objects_moved} objects moved significantly")
    
    def run(self):
        """Start the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closure."""
        try:
            if self.simulator and self.simulation_running:
                self.simulator.stop_simulation()
            
            if self.bridge:
                self.bridge.disconnect()
        except:
            pass  # Ignore cleanup errors
        
        self.root.destroy()


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Neophysics - Natural Language Physics Engine')
    parser.add_argument('--model', choices=['gpt-oss', 'gpt-oss-lora', 't5-small', 't5-trained'], default='auto',
                       help='Model backend to use (default: auto-detect)')
    args = parser.parse_args()
    
    app = InteractivePhysicsApp(model_override=args.model)
    app.run()


if __name__ == "__main__":
    main()
