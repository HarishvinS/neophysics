"""
Robust Data Generation Pipeline for the Learnable Physics Engine.

Generates complex, relational scenes and derives corresponding
natural language descriptions and target action sequences.
"""

import json
import argparse
import random
import numpy as np
from typing import List, Dict, Tuple, Any
import pybullet as p
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

from physics_engine import PhysicsEngine
from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject, ObjectType, MaterialType, Vector3


class RobustDataGenerator:
    """Generates high-quality text-scene-action sequence triplets."""

    def __init__(self):
        self.object_types = [ObjectType.SPHERE, ObjectType.BOX, ObjectType.RAMP]
        self.materials = list(MaterialType)
        self.colors = {
            "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
            "yellow": (1, 1, 0), "purple": (1, 0, 1), "cyan": (0, 1, 1)
        }
        self.color_names = list(self.colors.keys())
        self.relationships = ['on', 'next_to']
        self.scene_template_stats = {} # For quality reporting

    def _generate_random_object(self, object_id: str) -> DynamicPhysicsObject:
        """Generates a single object with random properties."""
        obj_type = random.choice(self.object_types)
        material = random.choice(self.materials)
        color_name = random.choice(self.color_names)
        color = self.colors[color_name]

        obj = DynamicPhysicsObject(
            object_id=object_id,
            object_type=obj_type,
            position=Vector3(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0.5, 2)),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(random.uniform(0.1, 0.5), random.uniform(0.1, 0.5), random.uniform(0.1, 0.5)),
            mass=random.uniform(0.5, 5.0),
            material=material,
            color=color
        )
        # Attach color_name as metadata after creation. This resolves the TypeError.
        obj.color_name = color_name
        # Attach initial_velocity after creation to resolve the TypeError.
        obj.initial_velocity = Vector3(0, 0, 0)

        if obj_type == ObjectType.RAMP:
            obj.scale = Vector3(random.uniform(1, 3), 0.1, random.uniform(1, 2))
            obj.rotation.y = random.uniform(-0.7, 0.7) # Angle in radians
            obj.mass = 0 # Ramps are static

        if obj_type == ObjectType.SPHERE:
            radius = random.uniform(0.1, 0.4)
            obj.scale = Vector3(radius, radius, radius)

        # Occasionally add initial velocity
        if random.random() < 0.2:
            vx = random.uniform(-3, 3)
            vy = random.uniform(-3, 3)
            vz = random.uniform(0, 2)
            obj.initial_velocity = Vector3(vx, vy, vz)

        return obj

    def _apply_relationship(self, subject: DynamicPhysicsObject, target: DynamicPhysicsObject, rel_type: str):
        """Adjusts subject's position based on a relationship to the target. Mimics the bridge logic."""
        if rel_type == 'on':
            target_top_z = target.position.z + (target.scale.z if target.object_type in [ObjectType.BOX, ObjectType.RAMP] else target.scale.x)
            subject_vertical_extent = subject.scale.z if subject.object_type in [ObjectType.BOX, ObjectType.RAMP] else subject.scale.x
            
            # Place on top, with slight random horizontal offset
            subject.position.x = target.position.x + random.uniform(-0.1, 0.1)
            subject.position.y = target.position.y + random.uniform(-0.1, 0.1)
            subject.position.z = target_top_z + subject_vertical_extent + 0.01

        elif rel_type == 'next_to':
            angle = random.uniform(0, 2 * np.pi)
            distance = (target.scale.x + subject.scale.x) + random.uniform(0.1, 0.5)
            subject.position.x = target.position.x + distance * np.cos(angle)
            subject.position.y = target.position.y + distance * np.sin(angle)
            subject.position.z = subject.scale.z # Place on ground

    def _generate_tower_scene(self, scene: DynamicPhysicsScene):
        """Generates a tower of stacked boxes, a more complex relational scene."""
        num_boxes = random.randint(2, 4)
        base_pos = Vector3(random.uniform(-2, 2), random.uniform(-2, 2), 0)
        base_scale = Vector3(random.uniform(0.4, 0.6), random.uniform(0.4, 0.6), random.uniform(0.1, 0.2))
        
        last_box = None
        for i in range(num_boxes):
            # Make boxes progressively smaller for stability
            scale_multiplier = 1 - (i * 0.15)
            box_scale = Vector3(base_scale.x * scale_multiplier, base_scale.y * scale_multiplier, base_scale.z)
            
            # Use the random object generator but override key properties for the tower
            box = self._generate_random_object(object_id=f"obj{i+1}")
            box.object_type = ObjectType.BOX
            box.scale = box_scale
            
            if last_box is None: # This is the base box
                box.position = Vector3(base_pos.x, base_pos.y, box.scale.z)
            else: # This is a stacked box
                self._apply_relationship(box, last_box, 'on')
                # Explicitly store the relationship for action sequence generation
                box.relationship = {'type': 'on', 'target_id': last_box.object_id}
            
            scene.add_object(box)
            last_box = box
        
        # Add a holistic description for the entire scene, which scene_to_text will use
        scene.scene_description = f"a tower of {num_boxes} {last_box.color_name} boxes"

    def generate_scene(self) -> DynamicPhysicsScene:
        """Generates a complete, potentially relational, physics scene."""
        scene = DynamicPhysicsScene(scene_id=f"scene_{random.randint(1000, 9999)}")
        
        # Occasionally generate a more structured "tower" scene
        if random.random() < 0.25:
            self._generate_tower_scene(scene)
        else:
            # Default behavior: generate a few random objects with a single relationship
            num_objects = random.randint(1, 4)
            objects = [self._generate_random_object(object_id=f"obj{i+1}") for i in range(num_objects)]

            if num_objects > 1 and random.random() < 0.6:
                subject = random.choice(objects)
                possible_targets = [o for o in objects if o.object_id != subject.object_id]
                if possible_targets:
                    target = random.choice(possible_targets)
                    rel_type = random.choice(self.relationships)
                    self._apply_relationship(subject, target, rel_type)
                    subject.relationship = {'type': rel_type, 'target_id': target.object_id}

            for obj in objects:
                scene.add_object(obj)
            
        return scene

    def scene_to_text(self, scene: DynamicPhysicsScene) -> str:
        """Generates a natural language description from a scene."""
        # Handle special, holistic scene descriptions first
        if hasattr(scene, 'scene_description') and scene.scene_description:
            prefix = random.choice(["Create ", "Make ", "Build ", "Add ", "Generate "])
            return prefix + scene.scene_description + "."

        descriptions = []
        
        # Sort objects to handle relationships correctly
        sorted_objects = sorted(scene.objects.values(), key=lambda o: hasattr(o, 'relationship'))

        for obj in sorted_objects:
            # Describe object with relationship
            if hasattr(obj, 'relationship'):
                rel = obj.relationship
                target_obj = scene.objects[rel['target_id']]
                
                # e.g., "a red sphere on the blue box"
                desc = f"a {obj.color_name} {obj.object_type.value} {rel['type']} the {target_obj.color_name} {target_obj.object_type.value}"
                descriptions.append(desc)
            
            # Describe standalone object
            else:
                # e.g., "a heavy green box"
                mass_desc = "heavy" if obj.mass > 3.0 else "light" if obj.mass < 1.0 else ""
                desc = f"a {mass_desc} {obj.color_name} {obj.object_type.value}".replace(" a ", " a ").strip()
                descriptions.append(desc)

            # Describe velocity
            if np.linalg.norm(obj.initial_velocity.to_tuple()) > 0.1:
                descriptions[-1] += " moving quickly"

        # Combine descriptions
        if not descriptions:
            return "Create an empty scene."
            
        # Use different joining phrases
        if len(descriptions) > 1:
            # Find the object that was a target of a relationship, if any
            rel_targets = {o.relationship['target_id'] for o in sorted_objects if hasattr(o, 'relationship')}
            # Filter out descriptions of objects that were only targets
            final_descs = [d for o, d in zip(sorted_objects, descriptions) if o.object_id not in rel_targets]
            
            if len(final_descs) > 1:
                text = " and ".join(final_descs)
            else:
                text = final_descs[0]
        else:
            text = descriptions[0]

        # Add a command prefix
        prefix = random.choice(["Create ", "Make ", "Build ", "Add ", "Generate "])
        return prefix + text + "."

    def scene_to_action_sequence(self, scene: DynamicPhysicsScene) -> str:
        """Generates the target action sequence string for the model."""
        actions = []
        relationships = []

        for obj in scene.objects.values():
            # Use a list to build the parts of the action string
            action_parts = [
                f"id={obj.object_id}",
                f"type={obj.object_type.value}",
                # Format tuples with fixed precision for consistency
                f"pos=({obj.position.x:.4f}, {obj.position.y:.4f}, {obj.position.z:.4f})",
                f"rot=({obj.rotation.x:.4f}, {obj.rotation.y:.4f}, {obj.rotation.z:.4f})",
                f"scale=({obj.scale.x:.4f}, {obj.scale.y:.4f}, {obj.scale.z:.4f})",
                f"mass={obj.mass:.2f}",
                f"material={obj.material.value}"
            ]

            # Add velocity only if it's significant, and ensure it's a standard float
            if hasattr(obj, 'initial_velocity') and np.linalg.norm([obj.initial_velocity.x, obj.initial_velocity.y, obj.initial_velocity.z]) > 1e-6:
                vel_str = f"({float(obj.initial_velocity.x):.4f}, {float(obj.initial_velocity.y):.4f}, {float(obj.initial_velocity.z):.4f})"
                action_parts.append(f"velocity={vel_str}")

            actions.append(" ".join(action_parts) + ";")
            if hasattr(obj, 'relationship'):
                rel = obj.relationship
                relationships.append(f"RELATE subject_id={obj.object_id} type={rel['type']} target_id={rel['target_id']};")

        # Relationships are specified after all objects are created
        return " ".join(actions) + " " + " ".join(relationships)

    def generate_dataset(self, num_examples: int) -> List[Dict]:
        """Generates a full dataset and saves it to a file."""
        dataset = []
        print(f"Generating {num_examples} examples...")
        for i in range(num_examples):
            scene = self.generate_scene()
            text = self.scene_to_text(scene)
            action_sequence = self.scene_to_action_sequence(scene)

            dataset.append({
                "id": f"sample_{i}",
                "text": text,
                "action_sequence": action_sequence,
            })
        
        # Save to file
        filepath = "training_data.json"
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Dataset with {num_examples} examples saved to {filepath}")
        return dataset

    def _create_interactive_gui(self, root: tk.Tk, engine: PhysicsEngine):
        """Creates the Tkinter GUI for the interactive mode."""

        class SceneController:
            def __init__(self, master: tk.Tk, engine: PhysicsEngine, generator: 'RobustDataGenerator'):
                self.master = master
                self.engine = engine
                self.generator = generator
                self.selected_body_id: int | None = None
                self.object_map: Dict[str, int] = {}

                # --- GUI Setup ---
                master.grid_columnconfigure(1, weight=1)
                master.grid_rowconfigure(0, weight=1)

                # --- Control Panel (Left) ---
                control_frame = ttk.Frame(master, padding="10")
                control_frame.grid(row=0, column=0, sticky="nsw")

                ttk.Label(control_frame, text="Commands", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=5, sticky="w")
                ttk.Button(control_frame, text="Add Sphere", command=lambda: self.add_object('sphere')).grid(row=1, column=0, sticky="ew", pady=2)
                ttk.Button(control_frame, text="Add Box", command=lambda: self.add_object('box')).grid(row=2, column=0, sticky="ew", pady=2)
                ttk.Button(control_frame, text="Add Ramp", command=lambda: self.add_object('ramp')).grid(row=3, column=0, sticky="ew", pady=2)
                
                ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)

                ttk.Button(control_frame, text="▶ Play Simulation", command=self.play_simulation).grid(row=5, column=0, sticky="ew", pady=2)
                ttk.Button(control_frame, text="💾 Save Scene", command=self.save_scene).grid(row=6, column=0, sticky="ew", pady=2)
                ttk.Button(control_frame, text="🔄 Reset Scene", command=self.reset_scene).grid(row=7, column=0, sticky="ew", pady=2)

                # --- Object Panel (Right) ---
                object_frame = ttk.Frame(master, padding="10")
                object_frame.grid(row=0, column=1, sticky="nsew")
                object_frame.grid_rowconfigure(1, weight=1)
                object_frame.grid_columnconfigure(0, weight=1)

                ttk.Label(object_frame, text="Scene Objects", font=("", 10, "bold")).grid(row=0, column=0, pady=5, sticky="w")
                
                list_frame = ttk.Frame(object_frame)
                list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
                list_frame.grid_rowconfigure(0, weight=1)
                list_frame.grid_columnconfigure(0, weight=1)

                self.object_listbox = tk.Listbox(list_frame, exportselection=False)
                self.object_listbox.grid(row=0, column=0, sticky="nsew")
                self.object_listbox.bind('<<ListboxSelect>>', self.on_object_select)
                
                scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.object_listbox.yview)
                scrollbar.grid(row=0, column=1, sticky="ns")
                self.object_listbox.config(yscrollcommand=scrollbar.set)

                # --- Properties Frame ---
                props_frame = ttk.Frame(object_frame)
                props_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
                props_frame.grid_columnconfigure(1, weight=1)

                ttk.Label(props_frame, text="Position (X,Y,Z):").grid(row=0, column=0, sticky="w")
                self.pos_x_var, self.pos_y_var, self.pos_z_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
                ttk.Entry(props_frame, textvariable=self.pos_x_var, width=8).grid(row=0, column=1, sticky="ew", padx=(5,2))
                ttk.Entry(props_frame, textvariable=self.pos_y_var, width=8).grid(row=0, column=2, sticky="ew", padx=2)
                ttk.Entry(props_frame, textvariable=self.pos_z_var, width=8).grid(row=0, column=3, sticky="ew", padx=2)

                ttk.Label(props_frame, text="Velocity (Vx,Vy,Vz):").grid(row=1, column=0, sticky="w")
                self.vel_x_var, self.vel_y_var, self.vel_z_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
                ttk.Entry(props_frame, textvariable=self.vel_x_var, width=8).grid(row=1, column=1, sticky="ew", padx=(5,2))
                ttk.Entry(props_frame, textvariable=self.vel_y_var, width=8).grid(row=1, column=2, sticky="ew", padx=2)
                ttk.Entry(props_frame, textvariable=self.vel_z_var, width=8).grid(row=1, column=3, sticky="ew", padx=2)

                ttk.Button(props_frame, text="Update Properties", command=self.update_properties).grid(row=2, column=1, columnspan=3, sticky="ew", pady=5)
                ttk.Button(props_frame, text="Delete Selected", command=self.delete_selected).grid(row=3, column=1, columnspan=3, sticky="ew")

            def add_object(self, obj_type: str):
                # Place new objects at a predictable, non-random position.
                # Users can then move them with the GUI controls.
                if obj_type == 'ramp':
                    pos = (0, 0, 0)  # Ramps start on the ground
                else:
                    pos = (0, 0, 0.5)  # Other objects start slightly above the ground

                color = (random.random(), random.random(), random.random())
                if obj_type == 'sphere':
                    self.engine.create_sphere(position=pos, color=color, radius=0.2)
                elif obj_type == 'box':
                    self.engine.create_box(position=pos, color=color, half_extents=[0.2, 0.2, 0.2])
                elif obj_type == 'ramp':
                    self.engine.create_ramp(position=pos, color=color, angle=0.3)
                self.refresh_object_list()

            def refresh_object_list(self):
                self.object_listbox.delete(0, tk.END)
                self.object_map = {name: body_id for name, body_id in self.engine.objects.items() if name != 'ground'}
                for name in sorted(self.object_map.keys()):
                    self.object_listbox.insert(tk.END, name)
                self.clear_properties()

            def on_object_select(self, event: tk.Event):
                selection_indices = self.object_listbox.curselection()
                if not selection_indices:
                    self.selected_body_id = None
                    self.clear_properties()
                    return
                
                selected_name = self.object_listbox.get(selection_indices[0])
                self.selected_body_id = self.object_map.get(selected_name)
                if self.selected_body_id is not None:
                    pos, _ = p.getBasePositionAndOrientation(self.selected_body_id, physicsClientId=self.engine.physics_client)
                    self.pos_x_var.set(f"{pos[0]:.3f}")
                    self.pos_y_var.set(f"{pos[1]:.3f}")
                    self.pos_z_var.set(f"{pos[2]:.3f}")
                    vel, _ = p.getBaseVelocity(self.selected_body_id, physicsClientId=self.engine.physics_client)
                    self.vel_x_var.set(f"{vel[0]:.3f}")
                    self.vel_y_var.set(f"{vel[1]:.3f}")
                    self.vel_z_var.set(f"{vel[2]:.3f}")

            def update_properties(self):
                if self.selected_body_id is None:
                    messagebox.showwarning("No Selection", "Please select an object from the list to update.")
                    return
                try:
                    # Update position
                    new_x = float(self.pos_x_var.get())
                    new_y = float(self.pos_y_var.get())
                    new_z = float(self.pos_z_var.get())
                    _, orn = p.getBasePositionAndOrientation(self.selected_body_id, physicsClientId=self.engine.physics_client)
                    p.resetBasePositionAndOrientation(self.selected_body_id, [new_x, new_y, new_z], orn, physicsClientId=self.engine.physics_client)

                    # Update velocity
                    new_vx = float(self.vel_x_var.get())
                    new_vy = float(self.vel_y_var.get())
                    new_vz = float(self.vel_z_var.get())
                    p.resetBaseVelocity(self.selected_body_id, [new_vx, new_vy, new_vz], physicsClientId=self.engine.physics_client)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Position and Velocity values must be valid numbers.")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not update properties: {e}")

            def delete_selected(self):
                if self.selected_body_id is None:
                    messagebox.showwarning("No Selection", "Please select an object from the list to delete.")
                    return
                
                # Find the name associated with the body_id to remove from engine.objects
                name_to_delete = None
                for name, body_id in self.engine.objects.items():
                    if body_id == self.selected_body_id:
                        name_to_delete = name
                        break
                
                if name_to_delete:
                    p.removeBody(self.selected_body_id, physicsClientId=self.engine.physics_client)
                    del self.engine.objects[name_to_delete]
                    if self.selected_body_id in self.engine.object_metadata:
                        del self.engine.object_metadata[self.selected_body_id]
                    
                    self.selected_body_id = None
                    self.refresh_object_list()

            def play_simulation(self):
                def step_and_reschedule(step_count):
                    if step_count > 0:
                        self.engine.step_simulation()
                        self.master.after(4, lambda: step_and_reschedule(step_count - 1)) # ~250Hz
                    else:
                        print("Simulation finished.")
                
                print("Playing simulation for 5 seconds...")
                step_and_reschedule(5 * 240)

            def save_scene(self):
                if not self.object_map:
                    messagebox.showinfo("Empty Scene", "Scene is empty. Add some objects before saving.")
                    return
                
                description = simpledialog.askstring("Save Scene", "Describe the scene in natural language:", parent=self.master)
                if not description or not description.strip():
                    messagebox.showwarning("Save Aborted", "Description cannot be empty.")
                    return

                scene = self.generator._create_scene_from_simulation(self.engine, list(self.object_map.values()))
                action_sequence = self.generator.scene_to_action_sequence(scene)
                self.generator._save_sample({"id": scene.scene_id, "text": description, "action_sequence": action_sequence})
                messagebox.showinfo("Success", "Scene saved successfully to training_data.json!")

            def reset_scene(self):
                self.engine.clear_objects(keep_ground=True)
                self.refresh_object_list()
                print("Scene reset.")

            def clear_properties(self):
                self.pos_x_var.set("")
                self.pos_y_var.set("")
                self.pos_z_var.set("")
                self.vel_x_var.set("")
                self.vel_y_var.set("")
                self.vel_z_var.set("")

        return SceneController(root, engine, self)

    def run_interactive_mode(self):
        """Runs an interactive session to manually create and label a scene."""
        print("\n--- Interactive Scene Creation ---")
        print("An interactive PyBullet window has opened.")
        print("Use the 'Object Controls' window to build and save your scene.")
        print("In the PyBullet window, you can use Ctrl+RightClick to drag objects.")
        print("------------------------------------")

        root = tk.Tk()
        root.title("Neophysics Scene Builder")
        
        engine = PhysicsEngine(use_gui=True)
        self._create_interactive_gui(root, engine)

        def on_closing():
            engine.disconnect()
            root.destroy()
            print("Exited interactive mode.")

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    def _create_scene_from_simulation(self, engine: PhysicsEngine, body_ids: List[int]) -> DynamicPhysicsScene:
        """Converts the current state of a PyBullet simulation into a DynamicPhysicsScene."""
        scene = DynamicPhysicsScene(scene_id=f"interactive_{int(time.time())}")
        obj_counter = 1

        def get_color_name(rgb_tuple: Tuple[float, float, float], colors_map: Dict[str, Any]) -> str:
            min_dist = float('inf')
            name = 'unknown'
            rgb_array = np.array(rgb_tuple[:3])
            for color_name, color_val in colors_map.items():
                dist = np.linalg.norm(rgb_array - np.array(color_val))
                if dist < min_dist:
                    min_dist = dist
                    name = color_name
            return name

        for body_id in body_ids:
            state = engine.get_object_state(body_id)
            if not state or body_id not in engine.object_metadata:
                continue
            
            meta = engine.object_metadata[body_id]
            obj_type_enum = ObjectType(meta['type'])
            
            scale = Vector3(1,1,1) # Default
            if obj_type_enum == ObjectType.SPHERE:
                r = meta['radius']
                scale = Vector3(r, r, r)
            elif obj_type_enum == ObjectType.BOX:
                scale = Vector3(*meta['half_extents'])
            elif obj_type_enum == ObjectType.RAMP:
                scale = Vector3(*meta['size'])

            pos, orn_quat = p.getBasePositionAndOrientation(body_id, physicsClientId=engine.physics_client)
            orn_euler = p.getEulerFromQuaternion(orn_quat)

            dpo = DynamicPhysicsObject(
                object_id=f"obj{obj_counter}",
                object_type=obj_type_enum,
                position=Vector3(*pos), rotation=Vector3(*orn_euler), scale=scale,
                mass=meta['mass'], material=MaterialType.WOOD, color=meta['color']
            )
            dpo.color_name = get_color_name(meta['color'], self.colors)
            dpo.initial_velocity = Vector3(*state['velocity'])
            scene.add_object(dpo)
            obj_counter += 1
        return scene

    def visualize_scene_3d(self, scene: DynamicPhysicsScene, duration: float = 5.0):
        """Visualize a scene in 3D using PyBullet."""
        # Initialize PyBullet
        physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
        
        # Create ground
        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics_client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, 
                         basePosition=[0, 0, 0], physicsClientId=physics_client)
        
        # Create objects from scene
        for obj in scene.objects.values():
            if obj.object_type == ObjectType.SPHERE:
                radius = obj.scale.x
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=physics_client)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, 
                                                 rgbaColor=list(obj.color) + [1.0], physicsClientId=physics_client)
            elif obj.object_type == ObjectType.BOX:
                half_extents = [obj.scale.x, obj.scale.y, obj.scale.z]
                collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                                 rgbaColor=list(obj.color) + [1.0], physicsClientId=physics_client)
            elif obj.object_type == ObjectType.RAMP:
                half_extents = [obj.scale.x/2, obj.scale.y/2, obj.scale.z/2]
                collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                                 rgbaColor=list(obj.color) + [1.0], physicsClientId=physics_client)
            
            # Create body
            orientation = p.getQuaternionFromEuler([obj.rotation.x, obj.rotation.y, obj.rotation.z])
            body_id = p.createMultiBody(
                baseMass=obj.mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[obj.position.x, obj.position.y, obj.position.z],
                baseOrientation=orientation,
                physicsClientId=physics_client
            )
            
            # Apply initial velocity if present
            if hasattr(obj, 'initial_velocity') and np.linalg.norm(obj.initial_velocity.to_tuple()) > 0.1:
                p.resetBaseVelocity(body_id, 
                                   linearVelocity=[obj.initial_velocity.x, obj.initial_velocity.y, obj.initial_velocity.z],
                                   physicsClientId=physics_client)
        
        print(f"Visualizing scene '{scene.scene_id}' for {duration} seconds...")
        print("Close the PyBullet window or press Ctrl+C to stop early.")
        
        # Run simulation
        try:
            for _ in range(int(duration * 240)):
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
        except KeyboardInterrupt:
            print("Visualization stopped by user.")
        
        p.disconnect(physicsClientId=physics_client)

    def visualize_dataset_sample(self, dataset: List[Dict], sample_index: int = 0, duration: float = 5.0):
        """Visualize a specific sample from the dataset."""
        if sample_index >= len(dataset):
            print(f"Sample index {sample_index} out of range. Dataset has {len(dataset)} samples.")
            return
        
        sample = dataset[sample_index]
        print(f"\nVisualizing sample {sample_index}:")
        print(f"Text: {sample['text']}")
        print(f"Action Sequence: {sample['action_sequence']}")
        
        # Parse action sequence back to scene
        scene = self._parse_action_sequence_to_scene(sample['action_sequence'], f"sample_{sample_index}")
        self.visualize_scene_3d(scene, duration)

    def _parse_action_sequence_to_scene(self, action_sequence: str, scene_id: str) -> DynamicPhysicsScene:
        """Parse action sequence back into a scene for visualization."""
        scene = DynamicPhysicsScene(scene_id)
        actions = [s.strip() for s in action_sequence.split(';') if s.strip()]
        
        for action_str in actions:
            if action_str.startswith('CREATE'):
                # Parse CREATE action
                import re
                params = {}
                param_matches = re.finditer(r'(\w+)=(\([^)]*\)|[^\s]+)', action_str)
                for match in param_matches:
                    key, value = match.groups()
                    params[key] = value
                
                if 'id' in params and 'type' in params:
                    try:
                        obj = DynamicPhysicsObject(
                            object_id=params['id'],
                            object_type=ObjectType(params['type']),
                            position=Vector3(*eval(params.get('pos', '(0,0,1)'))),
                            rotation=Vector3(*eval(params.get('rot', '(0,0,0)'))),
                            scale=Vector3(*eval(params.get('scale', '(0.5,0.5,0.5)'))),
                            mass=float(params.get('mass', '1.0')),
                            material=MaterialType(params.get('material', 'wood')),
                            color=self.colors.get('red', (1, 0, 0))  # Default color
                        )
                        
                        # Add velocity if present
                        if 'velocity' in params:
                            obj.initial_velocity = Vector3(*eval(params['velocity']))
                        else:
                            obj.initial_velocity = Vector3(0, 0, 0)
                        
                        scene.add_object(obj)
                    except Exception as e:
                        print(f"Warning: Failed to parse object from {action_str}: {e}")
        
        return scene

    def _save_sample(self, sample: Dict):
        """Loads existing dataset, appends a new sample, and saves it."""
        filepath = "training_data.json"
        try:
            with open(filepath, 'r') as f:
                dataset = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            dataset = []
        
        dataset.append(sample)

        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✅ Scene saved to {filepath}!")

def summarize_dataset(dataset: List[Dict], generator: RobustDataGenerator = None):
    """Prints a statistical summary of the generated dataset."""
    print("\n--- Dataset Quality Summary ---")
    if not dataset:
        print("Dataset is empty.")
        return

    num_examples = len(dataset)
    object_counts = {t.value: 0 for t in ObjectType}
    relationship_counts = {'on': 0, 'next_to': 0}
    velocity_count = 0
    total_objects = 0

    for example in dataset:
        actions = example['action_sequence'].split(';')
        for action in actions:
            if "CREATE" in action:
                total_objects += 1
                for obj_type in object_counts:
                    if f"type={obj_type}" in action:
                        object_counts[obj_type] += 1
                if "velocity=" in action:
                    velocity_count += 1
            elif "RELATE" in action:
                for rel_type in relationship_counts:
                    if f"type={rel_type}" in action:
                        relationship_counts[rel_type] += 1
    
    print(f"Total Examples: {num_examples}")
    print(f"Total Objects: {total_objects}")
    print("\nObject Type Distribution:")
    for obj_type, count in object_counts.items():
        print(f"  - {obj_type.capitalize()}: {count} ({count/total_objects:.1%})")
    
    print("\nRelationship Distribution:")
    for rel_type, count in relationship_counts.items():
        print(f"  - {rel_type.capitalize()}: {count}")

    print(f"\nObjects with Initial Velocity: {velocity_count} ({velocity_count/total_objects:.1%})")
    print("---------------------------------\n")
    
    if generator and generator.scene_template_stats:
        print("Scene Template Distribution:")
        total_templates = sum(generator.scene_template_stats.values())
        for template, count in generator.scene_template_stats.items():
            print(f"  - {template}: {count} ({count/total_templates:.1%})")
        print("---------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for the Learnable Physics Engine.")
    parser.add_argument('--num_examples', type=int, default=100, help='Number of examples to generate.')
    parser.add_argument('--visualize', type=int, help='Visualize a specific sample index from the existing dataset.')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode to manually create and label scenes.')
    parser.add_argument('--duration', type=float, default=5.0, help='Visualization duration in seconds')
    args = parser.parse_args()

    generator = RobustDataGenerator()

    if args.interactive:
        generator.run_interactive_mode()
    elif args.visualize is not None:
        # Load existing dataset and visualize
        try:
            with open('training_data.json', 'r') as f:
                dataset = json.load(f)
            generator.visualize_dataset_sample(dataset, args.visualize, args.duration)
        except FileNotFoundError:
            print("No training_data.json found. Generate a dataset first with --num_examples or --interactive.")
    else:
        # Generate a dataset and print its summary
        dataset = generator.generate_dataset(num_examples=args.num_examples)
        summarize_dataset(dataset, generator)
        
        # Optionally visualize the first sample
        if len(dataset) > 0:
            print("\nTo visualize samples, run: `python src/generate_dataset.py --visualize 0`")
            print("To create data manually, run: `python src/generate_dataset.py --interactive`")
