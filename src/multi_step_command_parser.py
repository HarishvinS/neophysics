"""
Multi-Step Command Parser
Handles complex commands with multiple steps and sequences.
Enables natural multi-step instructions like 'create a ball, place it on a ramp, then add a box below'.
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject
from scene_representation import ObjectType, MaterialType, Vector3
from conversational_context import ConversationalContext
from generative_understanding import ConceptualReasoner


class CommandType(Enum):
    """Types of individual commands."""
    CREATE = "create"
    PLACE = "place"
    MOVE = "move"
    MODIFY = "modify"
    DELETE = "delete"
    SIMULATE = "simulate"
    QUERY = "query"
    WAIT = "wait"


class SequenceConnector(Enum):
    """Types of sequence connectors between commands."""
    THEN = "then"
    AND = "and"
    AFTER = "after"
    BEFORE = "before"
    WHILE = "while"
    IF = "if"
    UNLESS = "unless"


@dataclass
class CommandStep:
    """Represents a single step in a multi-step command."""
    step_id: str
    command_type: CommandType
    raw_text: str
    parsed_action: str
    target_objects: List[str]
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of steps this depends on
    conditions: List[str]    # Conditions that must be met
    confidence: float
    
    def to_dict(self):
        return {
            'step_id': self.step_id,
            'command_type': self.command_type.value,
            'raw_text': self.raw_text,
            'parsed_action': self.parsed_action,
            'target_objects': self.target_objects,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'conditions': self.conditions,
            'confidence': self.confidence
        }


@dataclass
class CommandSequence:
    """Represents a complete sequence of commands."""
    sequence_id: str
    original_text: str
    steps: List[CommandStep]
    execution_order: List[str]  # Step IDs in execution order
    overall_confidence: float
    estimated_duration: float
    
    def to_dict(self):
        return {
            'sequence_id': self.sequence_id,
            'original_text': self.original_text,
            'steps': [step.to_dict() for step in self.steps],
            'execution_order': self.execution_order,
            'overall_confidence': self.overall_confidence,
            'estimated_duration': self.estimated_duration
        }


class MultiStepCommandParser:
    """Parses complex multi-step commands into executable sequences."""
    
    def __init__(self, context: ConversationalContext = None):
        """Initialize multi-step command parser."""
        self.context = context or ConversationalContext()
        self.conceptual_reasoner = ConceptualReasoner()
        
        # Command parsing patterns
        self.command_patterns = self._build_command_patterns()
        self.sequence_patterns = self._build_sequence_patterns()
        self.reference_patterns = self._build_reference_patterns()
        
        # Execution state
        self.current_scene = DynamicPhysicsScene("multi_step_scene")
        self.created_objects = {}  # step_id -> object_id mapping
    
    def _build_command_patterns(self) -> Dict[CommandType, List[str]]:
        """Build patterns for recognizing different command types."""
        return {
            CommandType.CREATE: [
                r'create (?:a |an )?(.+)',
                r'make (?:a |an )?(.+)',
                r'add (?:a |an )?(.+)',
                r'build (?:a |an )?(.+)',
                r'generate (?:a |an )?(.+)'
            ],
            CommandType.PLACE: [
                r'place (.+?) (on|above|below|next to|beside|between) (.+)',
                r'put (.+?) (on|above|below|next to|beside|between) (.+)',
                r'position (.+?) (on|above|below|next to|beside|between) (.+)',
                r'set (.+?) (on|above|below|next to|beside|between) (.+)'
            ],
            CommandType.MOVE: [
                r'move (.+?) (to|towards|away from) (.+)',
                r'shift (.+?) (to|towards|away from) (.+)',
                r'relocate (.+?) (to|towards|away from) (.+)'
            ],
            CommandType.MODIFY: [
                r'change (.+?) to (.+)',
                r'modify (.+?) to (.+)',
                r'make (.+?) (.+)',
                r'turn (.+?) (.+)',
                r'set (.+?) to (.+)'
            ],
            CommandType.DELETE: [
                r'delete (.+)',
                r'remove (.+)',
                r'destroy (.+)',
                r'clear (.+)'
            ],
            CommandType.SIMULATE: [
                r'simulate',
                r'run (?:the )?simulation',
                r'start (?:the )?physics',
                r'execute',
                r'play'
            ],
            CommandType.QUERY: [
                r'what (?:will |would )?happen',
                r'predict (.+)',
                r'what (?:if|about) (.+)',
                r'how (?:will |would )(.+)',
                r'why (?:will |would )(.+)'
            ],
            CommandType.WAIT: [
                r'wait (?:for )?(.+)',
                r'pause (?:for )?(.+)',
                r'delay (?:for )?(.+)'
            ]
        }
    
    def _build_sequence_patterns(self) -> Dict[SequenceConnector, List[str]]:
        """Build patterns for recognizing sequence connectors."""
        return {
            SequenceConnector.THEN: [r'\bthen\b', r'\bnext\b', r'\bafter that\b'],
            SequenceConnector.AND: [r'\band\b', r'\balso\b', r'\bplus\b'],
            SequenceConnector.AFTER: [r'\bafter\b', r'\bonce\b', r'\bwhen\b'],
            SequenceConnector.BEFORE: [r'\bbefore\b', r'\bprior to\b'],
            SequenceConnector.WHILE: [r'\bwhile\b', r'\bduring\b', r'\bas\b'],
            SequenceConnector.IF: [r'\bif\b', r'\bwhen\b', r'\bshould\b'],
            SequenceConnector.UNLESS: [r'\bunless\b', r'\bexcept\b', r'\bwithout\b']
        }
    
    def _build_reference_patterns(self) -> List[str]:
        """Build patterns for recognizing object references."""
        return [
            r'\bit\b', r'\bthat\b', r'\bthis\b', r'\bthem\b',
            r'\bthe (.+)', r'\bsaid (.+)', r'\bthe same (.+)',
            r'\bthe first (.+)', r'\bthe second (.+)', r'\bthe last (.+)'
        ]
    
    def parse_command_sequence(self, command_text: str) -> CommandSequence:
        """Parse a complex command into a sequence of steps."""
        sequence_id = f"seq_{int(time.time() * 1000)}"
        
        # Step 1: Split command into individual steps
        raw_steps = self._split_into_steps(command_text)
        
        # Step 2: Parse each step
        parsed_steps = []
        for i, raw_step in enumerate(raw_steps):
            step = self._parse_single_step(raw_step, i, sequence_id)
            parsed_steps.append(step)
        
        # Step 3: Resolve references and dependencies
        self._resolve_references(parsed_steps)
        self._analyze_dependencies(parsed_steps)
        
        # Step 4: Determine execution order
        execution_order = self._determine_execution_order(parsed_steps)
        
        # Step 5: Calculate confidence and duration
        overall_confidence = self._calculate_overall_confidence(parsed_steps)
        estimated_duration = self._estimate_duration(parsed_steps)
        
        sequence = CommandSequence(
            sequence_id=sequence_id,
            original_text=command_text,
            steps=parsed_steps,
            execution_order=execution_order,
            overall_confidence=overall_confidence,
            estimated_duration=estimated_duration
        )
        
        return sequence
    
    def _split_into_steps(self, command_text: str) -> List[str]:
        """Split command text into individual steps."""
        # Handle common separators
        separators = [
            r',\s*then\s+',
            r',\s*and\s+then\s+',
            r',\s*after\s+that\s+',
            r',\s*next\s+',
            r';\s*',
            r'\.\s*then\s+',
            r'\.\s*next\s+',
            r',\s*and\s+(?=\w+\s+(?:a|an|the)\s+)',  # "and" before object creation
        ]
        
        # Split by separators
        steps = [command_text]
        for separator in separators:
            new_steps = []
            for step in steps:
                new_steps.extend(re.split(separator, step, flags=re.IGNORECASE))
            steps = new_steps
        
        # Clean up steps
        cleaned_steps = []
        for step in steps:
            step = step.strip()
            if step and len(step) > 2:  # Ignore very short fragments
                cleaned_steps.append(step)
        
        return cleaned_steps
    
    def _parse_single_step(self, raw_text: str, step_index: int, sequence_id: str) -> CommandStep:
        """Parse a single command step."""
        step_id = f"{sequence_id}_step_{step_index}"
        
        # Resolve references using context
        resolved_text = self.context.resolve_references(raw_text)
        
        # Determine command type and extract information
        command_type, parsed_action, target_objects, parameters = self._analyze_command(resolved_text)
        
        # Calculate confidence
        confidence = self._calculate_step_confidence(command_type, target_objects, parameters)
        
        step = CommandStep(
            step_id=step_id,
            command_type=command_type,
            raw_text=raw_text,
            parsed_action=parsed_action,
            target_objects=target_objects,
            parameters=parameters,
            dependencies=[],
            conditions=[],
            confidence=confidence
        )
        
        return step
    
    def _analyze_command(self, text: str) -> Tuple[CommandType, str, List[str], Dict[str, Any]]:
        """Analyze a command to determine type and extract information."""
        text_lower = text.lower().strip()
        
        # Try to match command patterns
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return self._extract_command_details(command_type, match, text)
        
        # Default to CREATE if no specific pattern matches but contains object words
        object_indicators = ['ball', 'box', 'ramp', 'sphere', 'cube', 'cylinder']
        if any(indicator in text_lower for indicator in object_indicators):
            return CommandType.CREATE, text, [text], {}
        
        # Default to QUERY for questions
        if any(word in text_lower for word in ['what', 'how', 'why', 'will', 'would']):
            return CommandType.QUERY, text, [], {}
        
        return CommandType.CREATE, text, [text], {}
    
    def _extract_command_details(self, command_type: CommandType, match: re.Match, 
                               original_text: str) -> Tuple[CommandType, str, List[str], Dict[str, Any]]:
        """Extract detailed information from a matched command pattern."""
        groups = match.groups()
        parameters = {}
        target_objects = []
        
        if command_type == CommandType.CREATE:
            object_description = groups[0] if groups else original_text
            target_objects = [object_description]
            parameters['object_description'] = object_description
            parsed_action = f"create {object_description}"
        
        elif command_type == CommandType.PLACE:
            if len(groups) >= 3:
                object_ref = groups[0]
                spatial_relation = groups[1]
                reference_object = groups[2]
                
                target_objects = [object_ref, reference_object]
                parameters['spatial_relation'] = spatial_relation
                parameters['reference_object'] = reference_object
                parsed_action = f"place {object_ref} {spatial_relation} {reference_object}"
            else:
                parsed_action = original_text
        
        elif command_type == CommandType.MOVE:
            if len(groups) >= 3:
                object_ref = groups[0]
                direction = groups[1]
                target = groups[2]
                
                target_objects = [object_ref]
                parameters['direction'] = direction
                parameters['target'] = target
                parsed_action = f"move {object_ref} {direction} {target}"
            else:
                parsed_action = original_text
        
        elif command_type == CommandType.MODIFY:
            if len(groups) >= 2:
                object_ref = groups[0]
                new_property = groups[1]
                
                target_objects = [object_ref]
                parameters['new_property'] = new_property
                parsed_action = f"modify {object_ref} to {new_property}"
            else:
                parsed_action = original_text
        
        else:
            parsed_action = original_text
            if groups:
                target_objects = list(groups)
        
        return command_type, parsed_action, target_objects, parameters
    
    def _resolve_references(self, steps: List[CommandStep]):
        """Resolve object references between steps."""
        created_objects = {}  # object_description -> step_id
        
        for step in steps:
            # Track objects created in this step
            if step.command_type == CommandType.CREATE:
                obj_desc = step.parameters.get('object_description', '')
                created_objects[obj_desc] = step.step_id
            
            # Resolve references in target objects
            resolved_targets = []
            for target in step.target_objects:
                resolved_target = self._resolve_object_reference(target, created_objects)
                resolved_targets.append(resolved_target)
            
            step.target_objects = resolved_targets
    
    def _resolve_object_reference(self, reference: str, created_objects: Dict[str, str]) -> str:
        """Resolve a single object reference."""
        ref_lower = reference.lower().strip()
        
        # Direct match
        if ref_lower in created_objects:
            return ref_lower
        
        # Pronoun resolution
        pronouns = ['it', 'that', 'this']
        if ref_lower in pronouns and created_objects:
            # Return the most recently created object
            return list(created_objects.keys())[-1]
        
        # Partial match
        for obj_desc in created_objects.keys():
            if any(word in obj_desc for word in ref_lower.split()):
                return obj_desc
        
        return reference  # Return original if no resolution found
    
    def _analyze_dependencies(self, steps: List[CommandStep]):
        """Analyze dependencies between steps."""
        for i, step in enumerate(steps):
            # Steps that reference objects from previous steps depend on those steps
            for target in step.target_objects:
                for j, prev_step in enumerate(steps[:i]):
                    if (prev_step.command_type == CommandType.CREATE and 
                        target in prev_step.parameters.get('object_description', '')):
                        step.dependencies.append(prev_step.step_id)
            
            # PLACE commands depend on CREATE commands for both objects
            if step.command_type == CommandType.PLACE:
                for prev_step in steps[:i]:
                    if prev_step.command_type == CommandType.CREATE:
                        step.dependencies.append(prev_step.step_id)
    
    def _determine_execution_order(self, steps: List[CommandStep]) -> List[str]:
        """Determine the optimal execution order considering dependencies."""
        # Simple topological sort based on dependencies
        executed = set()
        execution_order = []
        
        while len(execution_order) < len(steps):
            for step in steps:
                if (step.step_id not in executed and 
                    all(dep in executed for dep in step.dependencies)):
                    execution_order.append(step.step_id)
                    executed.add(step.step_id)
                    break
        
        return execution_order
    
    def _calculate_step_confidence(self, command_type: CommandType, 
                                 target_objects: List[str], parameters: Dict[str, Any]) -> float:
        """Calculate confidence for a single step."""
        base_confidence = 0.8
        
        # Adjust based on command type
        type_confidence = {
            CommandType.CREATE: 0.9,
            CommandType.PLACE: 0.8,
            CommandType.MOVE: 0.7,
            CommandType.MODIFY: 0.6,
            CommandType.DELETE: 0.9,
            CommandType.SIMULATE: 0.95,
            CommandType.QUERY: 0.7,
            CommandType.WAIT: 0.8
        }
        
        confidence = type_confidence.get(command_type, base_confidence)
        
        # Adjust based on object clarity
        if target_objects:
            avg_object_clarity = sum(len(obj.split()) for obj in target_objects) / len(target_objects)
            confidence *= min(1.0, 0.5 + avg_object_clarity * 0.1)
        
        return min(0.95, confidence)
    
    def _calculate_overall_confidence(self, steps: List[CommandStep]) -> float:
        """Calculate overall confidence for the sequence."""
        if not steps:
            return 0.0
        
        step_confidences = [step.confidence for step in steps]
        return sum(step_confidences) / len(step_confidences)
    
    def _estimate_duration(self, steps: List[CommandStep]) -> float:
        """Estimate execution duration for the sequence."""
        # Base durations for different command types (in seconds)
        base_durations = {
            CommandType.CREATE: 1.0,
            CommandType.PLACE: 0.5,
            CommandType.MOVE: 0.5,
            CommandType.MODIFY: 0.3,
            CommandType.DELETE: 0.2,
            CommandType.SIMULATE: 3.0,
            CommandType.QUERY: 0.1,
            CommandType.WAIT: 1.0
        }
        
        total_duration = 0.0
        for step in steps:
            base_duration = base_durations.get(step.command_type, 1.0)
            # Add complexity factor based on number of target objects
            complexity_factor = 1.0 + len(step.target_objects) * 0.2
            total_duration += base_duration * complexity_factor
        
        return total_duration


def test_multi_step_command_parser():
    """Test the multi-step command parser."""
    print("Testing Multi-Step Command Parser...")
    
    # Create parser with context
    context = ConversationalContext()
    parser = MultiStepCommandParser(context)
    
    # Test complex multi-step commands
    test_commands = [
        "create a red ball, then place it on a ramp",
        "make a wooden box, put a sphere on top of it, and then add a second ball",
        "build a ramp, create a ball above it, then simulate the physics",
        "add a heavy metal sphere, place it between two cubes, and predict what happens",
        "create a U-shaped curved ramp, then add a ball at the top, wait for 2 seconds, and run the simulation"
    ]
    
    print(f"✅ Testing {len(test_commands)} multi-step commands...")
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n🔍 Command {i}: '{command}'")
        
        # Parse the command sequence
        sequence = parser.parse_command_sequence(command)
        
        print(f"   Steps identified: {len(sequence.steps)}")
        print(f"   Overall confidence: {sequence.overall_confidence:.2f}")
        print(f"   Estimated duration: {sequence.estimated_duration:.1f}s")
        
        # Show individual steps
        for j, step in enumerate(sequence.steps, 1):
            print(f"   Step {j}: {step.command_type.value}")
            print(f"     Action: {step.parsed_action}")
            print(f"     Targets: {step.target_objects}")
            print(f"     Dependencies: {step.dependencies}")
            print(f"     Confidence: {step.confidence:.2f}")
        
        # Show execution order
        print(f"   Execution order: {' → '.join([f'Step {sequence.execution_order.index(step_id) + 1}' for step_id in sequence.execution_order])}")
    
    print(f"\n✅ Multi-step command parser test completed!")
    print(f"🎯 Key capabilities demonstrated:")
    print(f"   • Complex command decomposition")
    print(f"   • Reference resolution between steps")
    print(f"   • Dependency analysis and ordering")
    print(f"   • Confidence assessment for sequences")
    print(f"   • Duration estimation")


if __name__ == "__main__":
    test_multi_step_command_parser()
