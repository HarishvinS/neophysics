"""
Advanced Physics Materials
Add realistic material properties including friction, elasticity, density, and thermal properties.
Enables sophisticated physics simulations with realistic material behavior.
"""

import numpy as np
import pybullet as p
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from scene_representation import MaterialType, Vector3


class AdvancedMaterialType(Enum):
    """Extended material types with realistic properties."""
    # Metals
    STEEL = "steel"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    IRON = "iron"
    TITANIUM = "titanium"
    
    # Polymers
    RUBBER_SOFT = "rubber_soft"
    RUBBER_HARD = "rubber_hard"
    PLASTIC_RIGID = "plastic_rigid"
    PLASTIC_FLEXIBLE = "plastic_flexible"
    FOAM = "foam"
    
    # Natural materials
    WOOD_OAK = "wood_oak"
    WOOD_PINE = "wood_pine"
    BAMBOO = "bamboo"
    CORK = "cork"
    
    # Ceramics and glass
    CERAMIC = "ceramic"
    PORCELAIN = "porcelain"
    GLASS_TEMPERED = "glass_tempered"
    GLASS_REGULAR = "glass_regular"
    
    # Composites
    CARBON_FIBER = "carbon_fiber"
    FIBERGLASS = "fiberglass"
    CONCRETE = "concrete"
    
    # Special materials
    ICE = "ice"
    SAND = "sand"
    MUD = "mud"
    LIQUID_WATER = "liquid_water"
    LIQUID_OIL = "liquid_oil"


@dataclass
class MaterialProperties:
    """Comprehensive material properties for realistic physics."""
    # Basic properties
    density: float  # kg/m³
    friction_static: float  # Static friction coefficient
    friction_kinetic: float  # Kinetic friction coefficient
    restitution: float  # Bounce coefficient (0-1)
    
    # Mechanical properties
    youngs_modulus: float  # Elasticity (Pa)
    poisson_ratio: float  # Lateral strain ratio
    yield_strength: float  # Yield strength (Pa)
    ultimate_strength: float  # Ultimate tensile strength (Pa)
    
    # Thermal properties
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    thermal_expansion: float  # 1/K
    melting_point: float  # K
    
    # Electrical properties
    electrical_conductivity: float  # S/m
    dielectric_constant: float  # Relative permittivity
    
    # Visual properties
    color: Tuple[float, float, float, float]  # RGBA
    roughness: float  # Surface roughness (0-1)
    metallic: float  # Metallic factor (0-1)
    transparency: float  # Transparency (0-1)
    
    # Special properties
    is_brittle: bool  # Breaks under stress
    is_magnetic: bool  # Magnetic material
    is_conductive: bool  # Electrically conductive
    is_fluid: bool  # Fluid behavior
    
    def to_dict(self):
        return {
            'density': self.density,
            'friction_static': self.friction_static,
            'friction_kinetic': self.friction_kinetic,
            'restitution': self.restitution,
            'youngs_modulus': self.youngs_modulus,
            'poisson_ratio': self.poisson_ratio,
            'yield_strength': self.yield_strength,
            'ultimate_strength': self.ultimate_strength,
            'thermal_conductivity': self.thermal_conductivity,
            'specific_heat': self.specific_heat,
            'thermal_expansion': self.thermal_expansion,
            'melting_point': self.melting_point,
            'electrical_conductivity': self.electrical_conductivity,
            'dielectric_constant': self.dielectric_constant,
            'color': list(self.color),
            'roughness': self.roughness,
            'metallic': self.metallic,
            'transparency': self.transparency,
            'is_brittle': self.is_brittle,
            'is_magnetic': self.is_magnetic,
            'is_conductive': self.is_conductive,
            'is_fluid': self.is_fluid
        }


class AdvancedMaterialDatabase:
    """Database of realistic material properties."""
    
    def __init__(self):
        """Initialize material database with realistic properties."""
        self.materials = self._create_material_database()
    
    def _create_material_database(self) -> Dict[AdvancedMaterialType, MaterialProperties]:
        """Create comprehensive material database."""
        materials = {}
        
        # Metals
        materials[AdvancedMaterialType.STEEL] = MaterialProperties(
            density=7850.0,
            friction_static=0.74,
            friction_kinetic=0.57,
            restitution=0.15,
            youngs_modulus=200e9,
            poisson_ratio=0.27,
            yield_strength=250e6,
            ultimate_strength=400e6,
            thermal_conductivity=50.2,
            specific_heat=490,
            thermal_expansion=12e-6,
            melting_point=1811,
            electrical_conductivity=6.99e6,
            dielectric_constant=1.0,
            color=(0.7, 0.7, 0.8, 1.0),
            roughness=0.3,
            metallic=1.0,
            transparency=0.0,
            is_brittle=False,
            is_magnetic=True,
            is_conductive=True,
            is_fluid=False
        )
        
        materials[AdvancedMaterialType.ALUMINUM] = MaterialProperties(
            density=2700.0,
            friction_static=0.61,
            friction_kinetic=0.47,
            restitution=0.25,
            youngs_modulus=69e9,
            poisson_ratio=0.33,
            yield_strength=95e6,
            ultimate_strength=110e6,
            thermal_conductivity=237,
            specific_heat=897,
            thermal_expansion=23e-6,
            melting_point=933,
            electrical_conductivity=37.8e6,
            dielectric_constant=1.0,
            color=(0.8, 0.8, 0.9, 1.0),
            roughness=0.2,
            metallic=1.0,
            transparency=0.0,
            is_brittle=False,
            is_magnetic=False,
            is_conductive=True,
            is_fluid=False
        )
        
        # Polymers
        materials[AdvancedMaterialType.RUBBER_SOFT] = MaterialProperties(
            density=920.0,
            friction_static=1.16,
            friction_kinetic=0.85,
            restitution=0.85,
            youngs_modulus=0.01e9,
            poisson_ratio=0.49,
            yield_strength=2e6,
            ultimate_strength=25e6,
            thermal_conductivity=0.16,
            specific_heat=1900,
            thermal_expansion=200e-6,
            melting_point=453,
            electrical_conductivity=1e-15,
            dielectric_constant=3.0,
            color=(0.2, 0.2, 0.2, 1.0),
            roughness=0.8,
            metallic=0.0,
            transparency=0.0,
            is_brittle=False,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=False
        )
        
        materials[AdvancedMaterialType.PLASTIC_RIGID] = MaterialProperties(
            density=1200.0,
            friction_static=0.4,
            friction_kinetic=0.3,
            restitution=0.4,
            youngs_modulus=3e9,
            poisson_ratio=0.35,
            yield_strength=50e6,
            ultimate_strength=60e6,
            thermal_conductivity=0.2,
            specific_heat=1500,
            thermal_expansion=80e-6,
            melting_point=423,
            electrical_conductivity=1e-16,
            dielectric_constant=2.5,
            color=(0.8, 0.2, 0.2, 1.0),
            roughness=0.4,
            metallic=0.0,
            transparency=0.0,
            is_brittle=True,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=False
        )
        
        # Wood
        materials[AdvancedMaterialType.WOOD_OAK] = MaterialProperties(
            density=750.0,
            friction_static=0.54,
            friction_kinetic=0.32,
            restitution=0.3,
            youngs_modulus=11e9,
            poisson_ratio=0.3,
            yield_strength=40e6,
            ultimate_strength=90e6,
            thermal_conductivity=0.17,
            specific_heat=2400,
            thermal_expansion=5e-6,
            melting_point=573,  # Decomposition temperature
            electrical_conductivity=1e-16,
            dielectric_constant=2.0,
            color=(0.6, 0.4, 0.2, 1.0),
            roughness=0.7,
            metallic=0.0,
            transparency=0.0,
            is_brittle=False,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=False
        )
        
        # Glass
        materials[AdvancedMaterialType.GLASS_REGULAR] = MaterialProperties(
            density=2500.0,
            friction_static=0.94,
            friction_kinetic=0.4,
            restitution=0.1,
            youngs_modulus=70e9,
            poisson_ratio=0.22,
            yield_strength=50e6,
            ultimate_strength=50e6,
            thermal_conductivity=1.05,
            specific_heat=840,
            thermal_expansion=9e-6,
            melting_point=1973,
            electrical_conductivity=1e-15,
            dielectric_constant=6.0,
            color=(0.9, 0.9, 0.9, 0.7),
            roughness=0.1,
            metallic=0.0,
            transparency=0.8,
            is_brittle=True,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=False
        )
        
        # Special materials
        materials[AdvancedMaterialType.ICE] = MaterialProperties(
            density=917.0,
            friction_static=0.1,
            friction_kinetic=0.03,
            restitution=0.2,
            youngs_modulus=9e9,
            poisson_ratio=0.33,
            yield_strength=3e6,
            ultimate_strength=5e6,
            thermal_conductivity=2.2,
            specific_heat=2100,
            thermal_expansion=-51e-6,  # Negative expansion
            melting_point=273,
            electrical_conductivity=1e-10,
            dielectric_constant=3.2,
            color=(0.8, 0.9, 1.0, 0.8),
            roughness=0.1,
            metallic=0.0,
            transparency=0.6,
            is_brittle=True,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=False
        )
        
        materials[AdvancedMaterialType.LIQUID_WATER] = MaterialProperties(
            density=1000.0,
            friction_static=0.0,  # No static friction for liquids
            friction_kinetic=0.0,
            restitution=0.0,
            youngs_modulus=2.2e9,  # Bulk modulus
            poisson_ratio=0.5,
            yield_strength=0.0,  # Liquids don't yield
            ultimate_strength=0.0,
            thermal_conductivity=0.6,
            specific_heat=4186,
            thermal_expansion=214e-6,
            melting_point=273,
            electrical_conductivity=5.5e-6,
            dielectric_constant=81.0,
            color=(0.2, 0.4, 0.8, 0.6),
            roughness=0.0,
            metallic=0.0,
            transparency=0.7,
            is_brittle=False,
            is_magnetic=False,
            is_conductive=False,
            is_fluid=True
        )
        
        return materials
    
    def get_material(self, material_type: AdvancedMaterialType) -> MaterialProperties:
        """Get material properties."""
        return self.materials.get(material_type)
    
    def get_all_materials(self) -> Dict[AdvancedMaterialType, MaterialProperties]:
        """Get all materials."""
        return self.materials.copy()
    
    def find_materials_by_property(self, property_name: str, min_value: float = None, max_value: float = None) -> List[AdvancedMaterialType]:
        """Find materials by property range."""
        matching_materials = []
        
        for material_type, properties in self.materials.items():
            if hasattr(properties, property_name):
                value = getattr(properties, property_name)
                
                if min_value is not None and value < min_value:
                    continue
                if max_value is not None and value > max_value:
                    continue
                
                matching_materials.append(material_type)
        
        return matching_materials


class AdvancedPhysicsEngine:
    """Enhanced physics engine with advanced material support."""
    
    def __init__(self, physics_client_id: int = 0):
        """Initialize advanced physics engine."""
        self.physics_client = physics_client_id
        self.material_db = AdvancedMaterialDatabase()
        self.temperature = 293.15  # Room temperature in Kelvin
        self.applied_materials = {}  # Track materials applied to objects
    
    def apply_material_to_object(self, body_id: int, material_type: AdvancedMaterialType):
        """Apply advanced material properties to a physics object."""
        material = self.material_db.get_material(material_type)
        if not material:
            raise ValueError(f"Unknown material type: {material_type}")
        
        # Apply basic physics properties
        p.changeDynamics(
            body_id,
            -1,
            mass=self._calculate_mass(body_id, material.density),
            lateralFriction=material.friction_kinetic,
            restitution=material.restitution,
            physicsClientId=self.physics_client
        )
        
        # Apply visual properties
        self._apply_visual_properties(body_id, material)
        
        # Store material for advanced physics calculations
        self.applied_materials[body_id] = material_type
        
        print(f"Applied {material_type.value} to object {body_id}")
    
    def _calculate_mass(self, body_id: int, density: float) -> float:
        """Calculate mass based on object volume and material density."""
        # Get object's collision shape info
        collision_info = p.getCollisionShapeData(body_id, -1, physicsClientId=self.physics_client)
        
        if not collision_info:
            return 1.0  # Default mass
        
        # Estimate volume based on shape type
        shape_type = collision_info[0][2]
        dimensions = collision_info[0][3]
        
        if shape_type == p.GEOM_BOX:
            # Box volume
            volume = 8 * dimensions[0] * dimensions[1] * dimensions[2]  # dimensions are half-extents
        elif shape_type == p.GEOM_SPHERE:
            # Sphere volume
            radius = dimensions[0]
            volume = (4/3) * np.pi * radius**3
        elif shape_type == p.GEOM_CYLINDER:
            # Cylinder volume
            radius = dimensions[0]
            height = dimensions[1]
            volume = np.pi * radius**2 * height
        else:
            volume = 1.0  # Default volume
        
        return density * volume
    
    def _apply_visual_properties(self, body_id: int, material: MaterialProperties):
        """Apply visual properties to object."""
        # Change color
        p.changeVisualShape(
            body_id,
            -1,
            rgbaColor=material.color,
            physicsClientId=self.physics_client
        )
        
        # Note: PyBullet has limited support for advanced visual properties
        # In a more advanced engine, we would apply roughness, metallic, etc.
    
    def simulate_thermal_effects(self, duration: float):
        """Simulate thermal effects on materials."""
        for body_id, material_type in self.applied_materials.items():
            material = self.material_db.get_material(material_type)
            
            # Check if material should melt
            if self.temperature > material.melting_point:
                print(f"Object {body_id} ({material_type.value}) would melt at current temperature!")
                
                # In a real implementation, we might change the object's state
                if material.is_fluid:
                    self._convert_to_fluid(body_id)
            
            # Apply thermal expansion
            expansion_factor = material.thermal_expansion * (self.temperature - 293.15)
            if abs(expansion_factor) > 1e-6:  # Significant expansion
                self._apply_thermal_expansion(body_id, expansion_factor)
    
    def _convert_to_fluid(self, body_id: int):
        """Convert solid object to fluid behavior."""
        # Reduce friction to near zero
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=0.01,
            restitution=0.0,
            physicsClientId=self.physics_client
        )
        print(f"Object {body_id} converted to fluid behavior")
    
    def _apply_thermal_expansion(self, body_id: int, expansion_factor: float):
        """Apply thermal expansion to object."""
        # Get current scale
        visual_data = p.getVisualShapeData(body_id, physicsClientId=self.physics_client)
        if visual_data:
            # In a real implementation, we would scale the object
            print(f"Object {body_id} thermal expansion: {expansion_factor:.6f}")
    
    def calculate_material_interaction(self, body_a: int, body_b: int) -> Dict[str, float]:
        """Calculate interaction properties between two materials."""
        material_a_type = self.applied_materials.get(body_a)
        material_b_type = self.applied_materials.get(body_b)
        
        if not material_a_type or not material_b_type:
            return {'friction': 0.5, 'restitution': 0.3}
        
        material_a = self.material_db.get_material(material_a_type)
        material_b = self.material_db.get_material(material_b_type)
        
        # Calculate combined properties
        combined_friction = np.sqrt(material_a.friction_kinetic * material_b.friction_kinetic)
        combined_restitution = np.sqrt(material_a.restitution * material_b.restitution)
        
        return {
            'friction': combined_friction,
            'restitution': combined_restitution,
            'thermal_transfer': self._calculate_thermal_transfer(material_a, material_b),
            'electrical_interaction': self._calculate_electrical_interaction(material_a, material_b)
        }
    
    def _calculate_thermal_transfer(self, material_a: MaterialProperties, material_b: MaterialProperties) -> float:
        """Calculate thermal transfer rate between materials."""
        # Harmonic mean of thermal conductivities
        if material_a.thermal_conductivity == 0 or material_b.thermal_conductivity == 0:
            return 0.0
        
        return 2 * material_a.thermal_conductivity * material_b.thermal_conductivity / (
            material_a.thermal_conductivity + material_b.thermal_conductivity
        )
    
    def _calculate_electrical_interaction(self, material_a: MaterialProperties, material_b: MaterialProperties) -> float:
        """Calculate electrical interaction between materials."""
        # Simple model: both materials must be conductive for electrical interaction
        if material_a.is_conductive and material_b.is_conductive:
            return min(material_a.electrical_conductivity, material_b.electrical_conductivity)
        return 0.0
    
    def get_material_recommendations(self, desired_properties: Dict[str, float]) -> List[Tuple[AdvancedMaterialType, float]]:
        """Get material recommendations based on desired properties."""
        recommendations = []
        
        for material_type, material in self.material_db.get_all_materials().items():
            score = 0.0
            total_weight = 0.0
            
            for prop_name, desired_value in desired_properties.items():
                if hasattr(material, prop_name):
                    actual_value = getattr(material, prop_name)
                    
                    # Calculate similarity score (closer = better)
                    if desired_value != 0:
                        similarity = 1.0 - abs(actual_value - desired_value) / abs(desired_value)
                    else:
                        similarity = 1.0 if actual_value == 0 else 0.0
                    
                    score += max(0.0, similarity)
                    total_weight += 1.0
            
            if total_weight > 0:
                final_score = score / total_weight
                recommendations.append((material_type, final_score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations


def test_advanced_physics_materials():
    """Test the advanced physics materials system."""
    print("Testing Advanced Physics Materials...")
    
    # Initialize physics
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    
    # Create advanced physics engine
    engine = AdvancedPhysicsEngine(physics_client)
    
    # Test material database
    material_db = AdvancedMaterialDatabase()
    
    print(f"✅ Material database loaded with {len(material_db.get_all_materials())} materials")
    
    # Test material properties
    steel = material_db.get_material(AdvancedMaterialType.STEEL)
    rubber = material_db.get_material(AdvancedMaterialType.RUBBER_SOFT)
    
    print(f"✅ Steel properties: density={steel.density}, friction={steel.friction_kinetic}, restitution={steel.restitution}")
    print(f"✅ Rubber properties: density={rubber.density}, friction={rubber.friction_kinetic}, restitution={rubber.restitution}")
    
    # Create test objects
    box_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5]),
        basePosition=[0, 0, 1],
        physicsClientId=physics_client
    )
    
    sphere_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.3),
        basePosition=[2, 0, 1],
        physicsClientId=physics_client
    )
    
    # Apply materials
    engine.apply_material_to_object(box_id, AdvancedMaterialType.STEEL)
    engine.apply_material_to_object(sphere_id, AdvancedMaterialType.RUBBER_SOFT)
    
    print(f"✅ Applied materials to objects")
    
    # Test material interactions
    interaction = engine.calculate_material_interaction(box_id, sphere_id)
    print(f"✅ Material interaction: {interaction}")
    
    # Test material search
    high_friction_materials = material_db.find_materials_by_property('friction_kinetic', min_value=0.8)
    print(f"✅ High friction materials: {[m.value for m in high_friction_materials]}")
    
    # Test material recommendations
    desired_props = {'density': 1000.0, 'friction_kinetic': 0.7, 'restitution': 0.5}
    recommendations = engine.get_material_recommendations(desired_props)
    
    print(f"✅ Material recommendations for {desired_props}:")
    for material, score in recommendations:
        print(f"   {material.value}: {score:.3f}")
    
    # Test thermal simulation
    engine.temperature = 400.0  # High temperature
    engine.simulate_thermal_effects(1.0)
    
    # Cleanup
    p.disconnect(physicsClientId=physics_client)
    
    print("✅ Advanced physics materials test completed!")


if __name__ == "__main__":
    test_advanced_physics_materials()
