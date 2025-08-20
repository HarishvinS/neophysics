"""
Interactive Interface for Text-to-Physics System
Combines text input, ML prediction, and live physics simulation in a unified GUI.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
from typing import Optional, Dict

from ml_physics_bridge import MLPhysicsBridge
from realtime_simulator import RealTimeSimulator
from physics_validator import PhysicsValidator
from model_architecture import TextToSceneModel, ModelConfig
from natural_conversation_interface import NaturalConversationInterface, ConversationMode
from command_disambiguation import CommandDisambiguator
from multi_step_command_parser import CommandSequence, CommandType
from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject


class InteractivePhysicsApp:
    """Main interactive application for text-to-physics system."""
    
    def __init__(self):
        """Initialize the interactive application."""
        self.root = tk.Tk()
        self.root.title("🧪 Learnable Physics Engine - Interactive Mode")
        self.root.geometry("1000x700")
        
        # Components
        self.model = None
        self.bridge = None
        self.simulator = None
        self.validator = None
        self.conversation_interface = None
        self.disambiguator = None

        # State
        self.model_loaded = False
        self.physics_initialized = False
        self.simulation_running = False
        
        # Setup UI
        self.setup_ui()
        
        # Try to load model automatically
        self.load_model_async()
        
        print("🚀 Interactive Physics Engine initialized!")
    
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
            text="🧪 Learnable Physics Engine - Interactive Mode", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Status indicators
        status_frame = ttk.Frame(header_frame)
        status_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.model_status = ttk.Label(status_frame, text="🔴 Model: Not Loaded", font=('Arial', 9))
        self.model_status.grid(row=0, column=0, padx=(0, 10))
        
        self.physics_status = ttk.Label(status_frame, text="🔴 Physics: Not Ready", font=('Arial', 9))
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
            text="🚀 Execute", 
            command=self.execute_command,
            style='Accent.TButton'
        )
        self.execute_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.validate_btn = ttk.Button(
            button_frame, 
            text="🔍 Validate", 
            command=self.validate_command
        )
        self.validate_btn.grid(row=0, column=1, padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame, 
            text="🧹 Clear", 
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
        self.log_message("🚀 Welcome to the Interactive Physics Engine!")
        self.log_message("Loading ML model...")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add a message to the status log."""
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding
        if level == "ERROR":
            prefix = "❌"
        elif level == "SUCCESS":
            prefix = "✅"
        elif level == "WARNING":
            prefix = "⚠️"
        else:
            prefix = "ℹ️"
        
        log_entry = f"[{timestamp}] {prefix} {message}\n"
        
        self.status_log.insert(tk.END, log_entry)
        self.status_log.see(tk.END)
        self.root.update()
    
    def _update_status_indicators(self):
        """Update status indicators in header."""
        if self.model_loaded:
            self.model_status.config(text="🟢 Model: Loaded")
        else:
            self.model_status.config(text="🔴 Model: Not Loaded")
        
        if self.physics_initialized:
            self.physics_status.config(text="🟢 Physics: Ready")
        else:
            self.physics_status.config(text="🔴 Physics: Not Ready")
    
    def _load_model_async(self):
        """Load model in background thread."""
        def load_model():
            try:
                self.progress_var.set(20)
                
                # Try to load trained model
                model_path = "models/trained_model/final_model.pth"
                
                if os.path.exists(model_path):
                    self.log_message("Loading trained model...")
                    
                    config = ModelConfig()
                    self.model = TextToSceneModel(
                        hidden_size=config.hidden_size, 
                        max_objects=config.max_objects
                    )
                    
                    import torch
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    
                    self.log_message("✅ Trained model loaded successfully!", "SUCCESS")
                else:
                    self.log_message("No trained model found, using untrained model", "WARNING")
                    
                    config = ModelConfig()
                    self.model = TextToSceneModel(
                        hidden_size=config.hidden_size, 
                        max_objects=config.max_objects
                    )
                
                self.progress_var.set(60)
                
                # Initialize components
                self.bridge = MLPhysicsBridge(self.model, use_gui=True)
                self.simulator = RealTimeSimulator(self.bridge, fps=60)
                self.validator = PhysicsValidator(self.bridge, self.simulator) # Validator can still be used
                self.conversation_interface = NaturalConversationInterface()
                self.disambiguator = CommandDisambiguator(self.conversation_interface.context)
                
                self.progress_var.set(80)
                
                # Add simulator callbacks
                self.simulator.add_event_callback(self.on_simulation_event)
                
                self.progress_var.set(100)
                
                self.model_loaded = True
                self.log_message("🎉 System ready! You can now enter physics commands.", "SUCCESS")
                
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

                # 1. Check for ambiguity
                self.log_message("1. Checking for ambiguity...")
                disambiguation_response = self.disambiguator.generate_disambiguation_response(command)
                if disambiguation_response.ambiguities_detected and not disambiguation_response.can_proceed_with_assumptions:
                    self.log_message(f"Clarification needed: {disambiguation_response.primary_question}", "WARNING")
                    for suggestion in disambiguation_response.alternative_suggestions:
                        self.log_message(f"  Suggestion: {suggestion}")
                    return
                elif disambiguation_response.assumptions_made:
                    for assumption in disambiguation_response.assumptions_made:
                        self.log_message(f"Proceeding with assumption: {assumption}")

                # 2. Process with conversational interface
                self.log_message("2. Processing with conversational interface...")
                conv_response = self.conversation_interface.process_conversation_input(command)
                self.log_message(f"System says: {conv_response.content}")

                # 3. If it's a command, build the scene and simulate
                if self.conversation_interface.current_mode == ConversationMode.COMMAND_EXECUTION:
                    self.log_message("3. Building scene from command...")
                    sequence = self.conversation_interface.command_parser.parse_command_sequence(command)
                    scene = self._build_scene_from_sequence(sequence)

                    if not scene.get_object_count():
                        self.log_message("No objects were created from the command.", "WARNING")
                        return

                    self.log_message(f"4. Rendering {scene.get_object_count()} objects...")
                    self.bridge.scene_to_physics(scene)

                    self.log_message("5. Reasoning about physics outcomes...")
                    analysis = self.conversation_interface.physics_reasoner.analyze_and_predict(scene)
                    self.log_message(f"   ...{analysis['reasoning_summary']}")

                    duration = float(self.duration_var.get())
                    self.log_message(f"6. Running simulation for {duration}s...")
                    self.start_simulation(duration)
                else:
                    self.log_message("Command was conversational, no simulation to run.")
                    if conv_response.follow_up_suggestions:
                        for suggestion in conv_response.follow_up_suggestions:
                            self.log_message(f"  Suggestion: {suggestion}")
                
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
        
        self.log_message(f"🔍 Validating: '{command}'")
        
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

    def _build_scene_from_sequence(self, sequence: CommandSequence) -> DynamicPhysicsScene:
        """Builds a DynamicPhysicsScene from a parsed command sequence."""
        scene = DynamicPhysicsScene(f"scene_{sequence.sequence_id}")
        created_objects_map: Dict[str, DynamicPhysicsObject] = {}

        # Use the execution order determined by the parser
        for step_id in sequence.execution_order:
            step = next((s for s in sequence.steps if s.step_id == step_id), None)
            if not step:
                continue

            if step.command_type == CommandType.CREATE:
                desc = step.parameters.get('object_description')
                if not desc:
                    continue
                
                self.log_message(f"   - Understanding '{desc}'...")
                concept = self.conversation_interface.conceptual_reasoner.synthesize_object_concept(desc)
                
                # For now, place at a default position. A more advanced version would parse position.
                new_obj = self.conversation_interface.conceptual_reasoner.create_object_from_concept(concept)
                
                scene.add_object(new_obj)
                created_objects_map[desc] = new_obj
                self.log_message(f"     > Created '{new_obj.object_id}' (Confidence: {concept['confidence']:.2f})")

            elif step.command_type == CommandType.PLACE:
                if len(step.target_objects) >= 2:
                    subject_desc = step.target_objects[0]
                    ref_desc = step.target_objects[1]

                    # Find the created DynamicPhysicsObject instances
                    subject_obj = next((obj for desc, obj in created_objects_map.items() if subject_desc in desc), None)
                    ref_obj = next((obj for desc, obj in created_objects_map.items() if ref_desc in desc), None)

                    if subject_obj and ref_obj:
                        relation = step.parameters.get('spatial_relation')
                        self.log_message(f"   - Placing '{subject_obj.object_id}' {relation} '{ref_obj.object_id}'...")

                        # Simplified placement logic (a more advanced version would be in a dedicated resolver)
                        if relation in ['on', 'above', 'on top of']:
                            ref_size = ref_obj.scale
                            # Position the subject on top of the reference object
                            new_pos = DynamicPhysicsObject.Vector3(
                                ref_obj.position.x,
                                ref_obj.position.y,
                                ref_obj.position.z + ref_size.z / 2 + subject_obj.scale.z / 2 + 0.1
                            )
                            subject_obj.position = new_pos
                            self.log_message(f"     > Moved '{subject_obj.object_id}' to {new_pos.to_list()}")
                        else:
                            self.log_message(f"     > (Placement logic for '{relation}' not yet implemented in this demo)")
                    else:
                        self.log_message(f"   - (Skipping PLACE: could not find objects '{subject_desc}' or '{ref_desc}')")
                else:
                    self.log_message(f"   - (Skipping PLACE: not enough objects in command)")

        self.conversation_interface.current_scene = scene
        return scene

    def start_simulation(self, duration: float):
        """Helper to start the simulation."""
        self.simulation_running = True
        self.simulator.start_simulation(duration=duration, record=True)
    
    def clear_scene(self):
        """Clear the physics scene."""
        if self.bridge:
            self.bridge.clear_scene()
            self.log_message("🧹 Scene cleared")
    
    def on_simulation_event(self, event_type: str, data: dict):
        """Handle simulation events."""
        if event_type == "simulation_stopped":
            self.simulation_running = False
            elapsed = data.get('elapsed_time', 0)
            steps = data.get('total_steps', 0)
            self.log_message(f"✅ Simulation completed: {elapsed:.2f}s, {steps} steps", "SUCCESS")
            
            # Analyze motion
            if hasattr(self, 'simulator') and self.simulator:
                analysis = self.simulator.analyze_motion()
                if 'objects' in analysis:
                    objects_moved = sum(1 for obj in analysis['objects'].values() 
                                      if obj.get('total_displacement', 0) > 0.1)
                    self.log_message(f"📊 Analysis: {objects_moved} objects moved significantly")
    
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
    app = InteractivePhysicsApp()
    app.run()


if __name__ == "__main__":
    main()
