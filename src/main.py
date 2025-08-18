"""
Main application for the Learnable Physics Engine
Combines Tkinter GUI with PyBullet physics simulation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from physics_engine import PhysicsEngine


class CommandParser:
    """Handles parsing of natural language commands (placeholder)."""
    def parse(self, command: str, physics_engine: PhysicsEngine):
        """
        Parse a command and execute it on the physics engine.
        This will be replaced by an ML model.
        """
        command_lower = command.lower()
        
        if "ball" in command_lower or "sphere" in command_lower:
            physics_engine.create_sphere(position=[0, 0, 2], radius=0.1, mass=1.0, color=(1, 0, 0))
            return "✅ Added ball based on command"
        elif "ramp" in command_lower:
            physics_engine.create_ramp(position=[0, 0, 0], angle=0.3, color=(0, 0, 1))
            return "✅ Added ramp based on command"
        elif "box" in command_lower or "cube" in command_lower:
            physics_engine.create_box(position=[1, 0, 1], half_extents=[0.2, 0.2, 0.2], mass=0.5, color=(0, 1, 0))
            return "✅ Added box based on command"
        elif "clear" in command_lower or "reset" in command_lower:
            physics_engine.clear_objects(keep_ground=True)
            return "✅ Cleared scene based on command"
        else:
            return "⚠️ Command not recognized. Try: 'add ball', 'create ramp', 'add box', 'clear scene'"


class PhysicsEngineApp:
    """Main application class combining GUI and physics engine."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("Learnable Physics Engine v0.1")
        self.root.geometry("800x600")
        
        # Physics engine (will be initialized when needed)
        self.physics_engine = None
        self.simulation_running = False
        self.command_parser = CommandParser()
        
        # Setup UI
        self.setup_ui()
        
        print("Application initialized successfully!")
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Header
        header_label = ttk.Label(
            main_frame, 
            text="🧪 Learnable Physics Engine", 
            font=('Arial', 16, 'bold')
        )
        header_label.grid(row=0, column=0, pady=(0, 20))
        
        # Command input section
        self.setup_command_section(main_frame)
        
        # Manual controls section
        self.setup_manual_controls(main_frame)
        
        # Status and log section
        self.setup_status_section(main_frame)
    
    def setup_command_section(self, parent):
        """Setup the command input section."""
        # Command frame
        cmd_frame = ttk.LabelFrame(parent, text="Natural Language Commands", padding="10")
        cmd_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        cmd_frame.columnconfigure(0, weight=1)
        
        # Command input
        ttk.Label(cmd_frame, text="Enter physics command:").grid(row=0, column=0, sticky=tk.W)
        
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(
            cmd_frame, 
            textvariable=self.command_var, 
            font=('Arial', 11),
            width=60
        )
        self.command_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 10))
        self.command_entry.bind('<Return>', lambda e: self.execute_command())
        
        # Execute button
        self.execute_btn = ttk.Button(
            cmd_frame, 
            text="Execute Command", 
            command=self.execute_command,
            style='Accent.TButton'
        )
        self.execute_btn.grid(row=2, column=0, pady=(0, 5))
        
        # Example commands
        examples_text = "Examples: 'create a ball', 'add a ramp', 'place a 2kg sphere at position 1,0,2'"
        ttk.Label(cmd_frame, text=examples_text, font=('Arial', 9), foreground='gray').grid(
            row=3, column=0, sticky=tk.W
        )
    
    def setup_manual_controls(self, parent):
        """Setup manual control buttons for testing."""
        # Manual controls frame
        manual_frame = ttk.LabelFrame(parent, text="Manual Controls (for testing)", padding="10")
        manual_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Button frame
        btn_frame = ttk.Frame(manual_frame)
        btn_frame.grid(row=0, column=0)
        
        # Control buttons
        ttk.Button(btn_frame, text="Initialize Physics", command=self.init_physics).grid(
            row=0, column=0, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Add Ball", command=self.add_ball_manual).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(btn_frame, text="Add Ramp", command=self.add_ramp_manual).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(btn_frame, text="Add Box", command=self.add_box_manual).grid(
            row=0, column=3, padx=5
        )
        ttk.Button(btn_frame, text="Clear Scene", command=self.clear_scene).grid(
            row=0, column=4, padx=5
        )
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation).grid(
            row=0, column=5, padx=(5, 0)
        )
    
    def setup_status_section(self, parent):
        """Setup status and logging section."""
        # Status frame
        status_frame = ttk.LabelFrame(parent, text="Status & Activity Log", padding="10")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # Status log
        self.status_log = scrolledtext.ScrolledText(
            status_frame, 
            height=15, 
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.status_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial message
        self.log_message("Welcome to the Learnable Physics Engine!")
        self.log_message("Click 'Initialize Physics' to start, then use manual controls or natural language commands.")
    
    def log_message(self, message):
        """Add a message to the status log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.status_log.insert(tk.END, log_entry)
        self.status_log.see(tk.END)
        self.root.update()
    
    def init_physics(self):
        """Initialize the physics engine."""
        try:
            if self.physics_engine is None:
                self.log_message("Initializing physics engine...")
                self.physics_engine = PhysicsEngine(use_gui=True)
                self.log_message("✅ Physics engine initialized successfully!")
                self.log_message("PyBullet 3D viewer should now be open.")
            else:
                self.log_message("Physics engine already initialized.")
        except Exception as e:
            self.log_message(f"❌ Error initializing physics: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize physics engine: {str(e)}")
    
    def execute_command(self):
        """Execute a natural language command (placeholder for now)."""
        command = self.command_var.get().strip()
        if not command:
            return
        
        self.log_message(f"Command received: '{command}'")
        
        try:
            if self.physics_engine is None:
                self.log_message("⚠️ Please initialize physics engine first!")
                return
            
            # Delegate parsing to the command parser
            response_message = self.command_parser.parse(command, self.physics_engine)
            self.log_message(response_message)
            
            # Clear command entry
            self.command_var.set("")
            
        except Exception as e:
            self.log_message(f"❌ Error executing command: {str(e)}")
    
    def add_ball_manual(self):
        """Add a ball manually."""
        if self.physics_engine is None:
            self.log_message("⚠️ Please initialize physics engine first!")
            return
        
        try:
            ball_id = self.physics_engine.create_sphere(
                position=[0, 0, 2],
                radius=0.1,
                mass=1.0,
                color=(1, 0, 0)
            )
            self.log_message(f"✅ Added red ball (ID: {ball_id})")
        except Exception as e:
            self.log_message(f"❌ Error adding ball: {str(e)}")
    
    def add_ramp_manual(self):
        """Add a ramp manually."""
        if self.physics_engine is None:
            self.log_message("⚠️ Please initialize physics engine first!")
            return
        
        try:
            ramp_id = self.physics_engine.create_ramp(
                position=[0, 0, 0],
                angle=0.3,
                color=(0, 0, 1)
            )
            self.log_message(f"✅ Added blue ramp (ID: {ramp_id})")
        except Exception as e:
            self.log_message(f"❌ Error adding ramp: {str(e)}")
    
    def add_box_manual(self):
        """Add a box manually."""
        if self.physics_engine is None:
            self.log_message("⚠️ Please initialize physics engine first!")
            return
        
        try:
            box_id = self.physics_engine.create_box(
                position=[1, 0, 1],
                half_extents=[0.2, 0.2, 0.2],
                mass=0.5,
                color=(0, 1, 0)
            )
            self.log_message(f"✅ Added green box (ID: {box_id})")
        except Exception as e:
            self.log_message(f"❌ Error adding box: {str(e)}")
    
    def clear_scene(self):
        """Clear all objects from the scene."""
        if self.physics_engine is None:
            self.log_message("⚠️ Please initialize physics engine first!")
            return
        
        try:
            self.physics_engine.clear_objects(keep_ground=True)
            self.log_message("✅ Scene cleared")
        except Exception as e:
            self.log_message(f"❌ Error clearing scene: {str(e)}")
    
    def run_simulation(self):
        """Run physics simulation for a few seconds."""
        if self.physics_engine is None:
            self.log_message("⚠️ Please initialize physics engine first!")
            return
        
        if self.simulation_running:
            self.log_message("⚠️ Simulation already running!")
            return
        
        def simulate():
            try:
                self.simulation_running = True
                self.log_message("🏃 Running simulation for 3 seconds...")
                self.physics_engine.run_simulation(duration=3.0, real_time=True)
                self.log_message("✅ Simulation completed")
            except Exception as e:
                self.log_message(f"❌ Error during simulation: {str(e)}")
            finally:
                self.simulation_running = False
        
        # Run simulation in separate thread to avoid blocking GUI
        sim_thread = threading.Thread(target=simulate)
        sim_thread.daemon = True
        sim_thread.start()
    
    def run(self):
        """Start the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closure."""
        try:
            if self.physics_engine is not None:
                self.physics_engine.disconnect()
        except (AttributeError, RuntimeError) as e:
            print(f"Warning during cleanup: {e}")
        
        self.root.destroy()


def main():
    """Main entry point."""
    app = PhysicsEngineApp()
    app.run()


if __name__ == "__main__":
    main()
