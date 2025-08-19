"""
Command Disambiguation System
Handles ambiguous commands by asking clarifying questions and providing suggestions.
Enables intelligent interaction when user intent is unclear or multiple interpretations are possible.
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from conversational_context import ConversationalContext
from generative_understanding import ConceptualReasoner
from multi_step_command_parser import MultiStepCommandParser, CommandType


class AmbiguityType(Enum):
    """Types of ambiguity that can occur in commands."""
    UNCLEAR_OBJECT = "unclear_object"
    MISSING_PARAMETERS = "missing_parameters"
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"
    INCOMPLETE_COMMAND = "incomplete_command"
    CONFLICTING_INSTRUCTIONS = "conflicting_instructions"
    UNKNOWN_CONCEPT = "unknown_concept"
    SPATIAL_AMBIGUITY = "spatial_ambiguity"


@dataclass
class AmbiguityDetection:
    """Represents detected ambiguity in a command."""
    ambiguity_type: AmbiguityType
    confidence: float
    problematic_text: str
    possible_interpretations: List[str]
    clarifying_questions: List[str]
    suggestions: List[str]
    context_needed: List[str]
    
    def to_dict(self):
        return {
            'ambiguity_type': self.ambiguity_type.value,
            'confidence': self.confidence,
            'problematic_text': self.problematic_text,
            'possible_interpretations': self.possible_interpretations,
            'clarifying_questions': self.clarifying_questions,
            'suggestions': self.suggestions,
            'context_needed': self.context_needed
        }


@dataclass
class DisambiguationResponse:
    """Response to help disambiguate a command."""
    response_id: str
    original_command: str
    ambiguities_detected: List[AmbiguityDetection]
    primary_question: str
    alternative_suggestions: List[str]
    confidence_threshold: float
    can_proceed_with_assumptions: bool
    assumptions_made: List[str]
    
    def to_dict(self):
        return {
            'response_id': self.response_id,
            'original_command': self.original_command,
            'ambiguities_detected': [amb.to_dict() for amb in self.ambiguities_detected],
            'primary_question': self.primary_question,
            'alternative_suggestions': self.alternative_suggestions,
            'confidence_threshold': self.confidence_threshold,
            'can_proceed_with_assumptions': self.can_proceed_with_assumptions,
            'assumptions_made': self.assumptions_made
        }


class CommandDisambiguator:
    """Handles command disambiguation through intelligent questioning and suggestions."""
    
    def __init__(self, context: ConversationalContext = None):
        """Initialize command disambiguator."""
        self.context = context or ConversationalContext()
        self.conceptual_reasoner = ConceptualReasoner()
        self.command_parser = MultiStepCommandParser(self.context)
        
        # Ambiguity detection patterns
        self.ambiguity_patterns = self._build_ambiguity_patterns()
        self.clarification_templates = self._build_clarification_templates()
        self.suggestion_generators = self._build_suggestion_generators()
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.7
        self.disambiguation_threshold = 0.5
    
    def _build_ambiguity_patterns(self) -> Dict[AmbiguityType, List[Dict[str, Any]]]:
        """Build patterns for detecting different types of ambiguity."""
        return {
            AmbiguityType.UNCLEAR_OBJECT: [
                {
                    'pattern': r'\b(thing|object|item|stuff)\b',
                    'confidence': 0.9,
                    'description': 'Generic object reference without specific type'
                },
                {
                    'pattern': r'\b(it|that|this)\b(?!\s+(?:ball|box|sphere|cube|ramp))',
                    'confidence': 0.8,
                    'description': 'Pronoun without clear antecedent'
                }
            ],
            AmbiguityType.MISSING_PARAMETERS: [
                {
                    'pattern': r'\b(place|put|move)\s+\w+\s*$',
                    'confidence': 0.9,
                    'description': 'Placement command without destination'
                },
                {
                    'pattern': r'\b(create|make|add)\s+(?:a\s+)?(\w+)\s*$',
                    'confidence': 0.7,
                    'description': 'Creation command without properties or location'
                }
            ],
            AmbiguityType.MULTIPLE_INTERPRETATIONS: [
                {
                    'pattern': r'\b(ball|sphere)\b.*\b(ball|sphere)\b',
                    'confidence': 0.8,
                    'description': 'Multiple similar objects mentioned'
                },
                {
                    'pattern': r'\b(on|above|over)\b',
                    'confidence': 0.6,
                    'description': 'Spatial relationship could have multiple meanings'
                }
            ],
            AmbiguityType.INCOMPLETE_COMMAND: [
                {
                    'pattern': r'\b(and|then|after)\s*$',
                    'confidence': 0.9,
                    'description': 'Command ends with conjunction, suggesting more to come'
                },
                {
                    'pattern': r'^\s*(?:also|plus|additionally)\b',
                    'confidence': 0.8,
                    'description': 'Command starts with addition word, suggesting continuation'
                }
            ],
            AmbiguityType.UNKNOWN_CONCEPT: [
                {
                    'pattern': r'\b([a-z]+(?:-[a-z]+)*)\b',
                    'confidence': 0.5,
                    'description': 'Potentially unknown concept or object type'
                }
            ],
            AmbiguityType.SPATIAL_AMBIGUITY: [
                {
                    'pattern': r'\b(between|among|around|near)\b',
                    'confidence': 0.7,
                    'description': 'Spatial relationship requiring multiple reference objects'
                },
                {
                    'pattern': r'\b(left|right|front|back)\b(?!\s+(?:of|side))',
                    'confidence': 0.8,
                    'description': 'Relative direction without clear reference frame'
                }
            ]
        }
    
    def _build_clarification_templates(self) -> Dict[AmbiguityType, List[str]]:
        """Build templates for clarifying questions."""
        return {
            AmbiguityType.UNCLEAR_OBJECT: [
                "What type of object would you like me to create?",
                "Could you specify what kind of {object} you mean?",
                "I need more details about the object. What should it look like?"
            ],
            AmbiguityType.MISSING_PARAMETERS: [
                "Where would you like me to place the {object}?",
                "What properties should the {object} have (size, material, color)?",
                "Could you specify the location or target for this action?"
            ],
            AmbiguityType.MULTIPLE_INTERPRETATIONS: [
                "I see multiple ways to interpret this. Do you mean:",
                "Which {object} are you referring to?",
                "Could you clarify which interpretation you prefer?"
            ],
            AmbiguityType.INCOMPLETE_COMMAND: [
                "It seems like your command was cut off. What would you like me to do next?",
                "I'm waiting for the rest of your instruction. Please continue.",
                "What should happen after {action}?"
            ],
            AmbiguityType.CONFLICTING_INSTRUCTIONS: [
                "I notice conflicting instructions. Which should I prioritize?",
                "These actions seem to contradict each other. Could you clarify?",
                "How should I resolve this conflict in the instructions?"
            ],
            AmbiguityType.UNKNOWN_CONCEPT: [
                "I'm not familiar with '{concept}'. Could you describe what it should look like?",
                "'{concept}' is new to me. What properties should it have?",
                "Help me understand '{concept}' - what shape, size, or function should it have?"
            ],
            AmbiguityType.SPATIAL_AMBIGUITY: [
                "Could you be more specific about the spatial arrangement?",
                "Which objects should serve as reference points for '{relation}'?",
                "I need clearer spatial instructions. Could you elaborate?"
            ]
        }
    
    def _build_suggestion_generators(self) -> Dict[AmbiguityType, List[str]]:
        """Build suggestion generators for different ambiguity types."""
        return {
            AmbiguityType.UNCLEAR_OBJECT: [
                "Try: 'create a red ball' or 'make a wooden box'",
                "Specify object type: ball, box, ramp, sphere, cube, cylinder",
                "Add properties: 'large metal sphere' or 'small rubber ball'"
            ],
            AmbiguityType.MISSING_PARAMETERS: [
                "Add location: 'place the ball on the ramp'",
                "Specify properties: 'create a large red ball'",
                "Include destination: 'move the box to the left'"
            ],
            AmbiguityType.MULTIPLE_INTERPRETATIONS: [
                "Use specific references: 'the first ball' or 'the red sphere'",
                "Number your objects: 'ball 1' and 'ball 2'",
                "Add distinguishing properties: 'the large ball' vs 'the small ball'"
            ],
            AmbiguityType.INCOMPLETE_COMMAND: [
                "Complete your thought: 'create a ball, then...'",
                "Add the next step: 'place it on a ramp'",
                "Finish the sequence: 'and then simulate physics'"
            ],
            AmbiguityType.UNKNOWN_CONCEPT: [
                "Describe the shape: 'U-shaped curved object'",
                "Explain the function: 'aerodynamic wing for lift'",
                "Use familiar terms: 'like a ball but hollow'"
            ],
            AmbiguityType.SPATIAL_AMBIGUITY: [
                "Be more specific: 'between the red ball and blue box'",
                "Use clear references: 'to the left of the ramp'",
                "Specify exact positions: 'above the center of the platform'"
            ]
        }
    
    def analyze_command_ambiguity(self, command: str) -> List[AmbiguityDetection]:
        """Analyze a command for potential ambiguities."""
        ambiguities = []
        command_lower = command.lower()
        
        # Check each ambiguity type
        for ambiguity_type, patterns in self.ambiguity_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                matches = re.finditer(pattern, command_lower)
                
                for match in matches:
                    # Skip if this is a known concept for unknown concept detection
                    if ambiguity_type == AmbiguityType.UNKNOWN_CONCEPT:
                        matched_text = match.group(1) if match.groups() else match.group(0)
                        if self._is_known_concept(matched_text):
                            continue
                    
                    # Create ambiguity detection
                    ambiguity = self._create_ambiguity_detection(
                        ambiguity_type, pattern_info, match, command
                    )
                    
                    if ambiguity:
                        ambiguities.append(ambiguity)
        
        # Check for context-dependent ambiguities
        context_ambiguities = self._check_context_ambiguities(command)
        ambiguities.extend(context_ambiguities)
        
        # Remove duplicates and sort by confidence
        unique_ambiguities = self._deduplicate_ambiguities(ambiguities)
        unique_ambiguities.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_ambiguities
    
    def _create_ambiguity_detection(self, ambiguity_type: AmbiguityType, 
                                  pattern_info: Dict[str, Any], 
                                  match: re.Match, command: str) -> Optional[AmbiguityDetection]:
        """Create an ambiguity detection from a pattern match."""
        matched_text = match.group(0)
        
        # Generate possible interpretations
        interpretations = self._generate_interpretations(ambiguity_type, matched_text, command)
        
        # Generate clarifying questions
        questions = self._generate_clarifying_questions(ambiguity_type, matched_text)
        
        # Generate suggestions
        suggestions = self.suggestion_generators.get(ambiguity_type, [])
        
        # Determine context needed
        context_needed = self._determine_context_needed(ambiguity_type, matched_text)
        
        return AmbiguityDetection(
            ambiguity_type=ambiguity_type,
            confidence=pattern_info['confidence'],
            problematic_text=matched_text,
            possible_interpretations=interpretations,
            clarifying_questions=questions,
            suggestions=suggestions[:3],  # Limit to top 3 suggestions
            context_needed=context_needed
        )
    
    def _generate_interpretations(self, ambiguity_type: AmbiguityType, 
                                text: str, command: str) -> List[str]:
        """Generate possible interpretations for ambiguous text."""
        interpretations = []
        
        if ambiguity_type == AmbiguityType.UNCLEAR_OBJECT:
            if text in ['thing', 'object', 'item']:
                interpretations = ['ball', 'box', 'sphere', 'cube', 'ramp']
            elif text in ['it', 'that', 'this']:
                # Try to get from context
                recent_objects = self.context.get_current_entities()
                interpretations = [entity.name for entity in recent_objects.values() 
                                 if entity.entity_type == 'physics_object'][:3]
        
        elif ambiguity_type == AmbiguityType.MULTIPLE_INTERPRETATIONS:
            if 'on' in text:
                interpretations = ['directly on top of', 'touching the surface of', 'supported by']
            elif 'above' in text:
                interpretations = ['directly above', 'higher than', 'floating over']
        
        elif ambiguity_type == AmbiguityType.SPATIAL_AMBIGUITY:
            if 'between' in text:
                interpretations = ['equidistant from two objects', 'in the space separating objects']
            elif 'near' in text:
                interpretations = ['close to', 'adjacent to', 'within reach of']
        
        return interpretations[:3]  # Limit to top 3
    
    def _generate_clarifying_questions(self, ambiguity_type: AmbiguityType, text: str) -> List[str]:
        """Generate clarifying questions for ambiguous text."""
        templates = self.clarification_templates.get(ambiguity_type, [])
        questions = []
        
        for template in templates[:2]:  # Limit to 2 questions
            if '{object}' in template:
                question = template.format(object=text)
            elif '{concept}' in template:
                question = template.format(concept=text)
            elif '{action}' in template:
                question = template.format(action=text)
            elif '{relation}' in template:
                question = template.format(relation=text)
            else:
                question = template
            
            questions.append(question)
        
        return questions
    
    def _determine_context_needed(self, ambiguity_type: AmbiguityType, text: str) -> List[str]:
        """Determine what context information is needed."""
        context_needed = []
        
        if ambiguity_type == AmbiguityType.UNCLEAR_OBJECT:
            context_needed = ['object_type', 'object_properties']
        elif ambiguity_type == AmbiguityType.MISSING_PARAMETERS:
            context_needed = ['location', 'properties', 'target_object']
        elif ambiguity_type == AmbiguityType.SPATIAL_AMBIGUITY:
            context_needed = ['reference_objects', 'spatial_relationship']
        elif ambiguity_type == AmbiguityType.UNKNOWN_CONCEPT:
            context_needed = ['concept_description', 'visual_properties', 'function']
        
        return context_needed
    
    def _check_context_ambiguities(self, command: str) -> List[AmbiguityDetection]:
        """Check for ambiguities that depend on conversational context."""
        ambiguities = []
        
        # Check for pronouns without clear antecedents
        pronouns = re.finditer(r'\b(it|that|this|them|those|these)\b', command.lower())
        for match in pronouns:
            pronoun = match.group(0)
            
            # Check if we have context for this pronoun
            if not self._has_clear_antecedent(pronoun):
                ambiguity = AmbiguityDetection(
                    ambiguity_type=AmbiguityType.UNCLEAR_OBJECT,
                    confidence=0.8,
                    problematic_text=pronoun,
                    possible_interpretations=self._get_possible_antecedents(pronoun),
                    clarifying_questions=[f"What does '{pronoun}' refer to?"],
                    suggestions=["Use specific object names instead of pronouns"],
                    context_needed=['object_reference']
                )
                ambiguities.append(ambiguity)
        
        return ambiguities
    
    def _has_clear_antecedent(self, pronoun: str) -> bool:
        """Check if a pronoun has a clear antecedent in context."""
        recent_entities = self.context.get_current_entities()
        
        if pronoun in ['it', 'that', 'this']:
            # Need at least one recent object
            return len([e for e in recent_entities.values() if e.entity_type == 'physics_object']) > 0
        elif pronoun in ['them', 'those', 'these']:
            # Need multiple recent objects
            return len([e for e in recent_entities.values() if e.entity_type == 'physics_object']) > 1
        
        return False
    
    def _get_possible_antecedents(self, pronoun: str) -> List[str]:
        """Get possible antecedents for a pronoun from context."""
        recent_entities = self.context.get_current_entities()
        physics_objects = [e.name for e in recent_entities.values() if e.entity_type == 'physics_object']
        
        return physics_objects[:3]  # Return up to 3 possibilities
    
    def _is_known_concept(self, concept: str) -> bool:
        """Check if a concept is known to the system."""
        known_objects = [
            'ball', 'sphere', 'box', 'cube', 'ramp', 'cylinder', 'plane',
            'pendulum', 'spring', 'chain', 'rope', 'bridge', 'car', 'wheel'
        ]
        
        known_materials = [
            'wood', 'metal', 'rubber', 'plastic', 'glass', 'stone',
            'steel', 'aluminum', 'copper', 'iron'
        ]
        
        known_properties = [
            'red', 'blue', 'green', 'yellow', 'black', 'white',
            'large', 'small', 'heavy', 'light', 'fast', 'slow'
        ]
        
        return (concept.lower() in known_objects or 
                concept.lower() in known_materials or 
                concept.lower() in known_properties)
    
    def _deduplicate_ambiguities(self, ambiguities: List[AmbiguityDetection]) -> List[AmbiguityDetection]:
        """Remove duplicate ambiguity detections."""
        seen = set()
        unique = []
        
        for ambiguity in ambiguities:
            key = (ambiguity.ambiguity_type, ambiguity.problematic_text)
            if key not in seen:
                seen.add(key)
                unique.append(ambiguity)
        
        return unique
    
    def generate_disambiguation_response(self, command: str) -> DisambiguationResponse:
        """Generate a complete disambiguation response for a command."""
        response_id = f"disambig_{int(time.time() * 1000)}"
        
        # Analyze ambiguities
        ambiguities = self.analyze_command_ambiguity(command)
        
        # Determine if we can proceed with assumptions
        can_proceed = len(ambiguities) == 0 or all(amb.confidence < self.disambiguation_threshold for amb in ambiguities)
        
        # Generate primary question
        if ambiguities:
            primary_ambiguity = ambiguities[0]
            primary_question = primary_ambiguity.clarifying_questions[0] if primary_ambiguity.clarifying_questions else "Could you clarify your command?"
        else:
            primary_question = "Your command is clear. Should I proceed?"
        
        # Generate alternative suggestions
        suggestions = []
        for ambiguity in ambiguities[:2]:  # Top 2 ambiguities
            suggestions.extend(ambiguity.suggestions[:2])  # 2 suggestions each
        
        # Remove duplicates
        suggestions = list(dict.fromkeys(suggestions))[:4]  # Max 4 suggestions
        
        # Generate assumptions if proceeding
        assumptions = []
        if can_proceed and ambiguities:
            for ambiguity in ambiguities:
                if ambiguity.possible_interpretations:
                    assumption = f"Assuming '{ambiguity.problematic_text}' means '{ambiguity.possible_interpretations[0]}'"
                    assumptions.append(assumption)
        
        return DisambiguationResponse(
            response_id=response_id,
            original_command=command,
            ambiguities_detected=ambiguities,
            primary_question=primary_question,
            alternative_suggestions=suggestions,
            confidence_threshold=self.min_confidence_threshold,
            can_proceed_with_assumptions=can_proceed,
            assumptions_made=assumptions
        )


def test_command_disambiguation():
    """Test the command disambiguation system."""
    print("Testing Command Disambiguation System...")
    
    # Create disambiguator with context
    context = ConversationalContext()
    disambiguator = CommandDisambiguator(context)
    
    # Test ambiguous commands
    test_commands = [
        "create a thing",
        "place it on the ramp",
        "make a ball and put it",
        "add something between the objects",
        "create a U-shaped thingy",
        "move the ball to the left",
        "place the sphere above",
        "make a large",
        "create a ball, then",
        "put the red one near the blue"
    ]
    
    print(f"✅ Testing {len(test_commands)} ambiguous commands...")
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n🔍 Command {i}: '{command}'")
        
        # Analyze ambiguities
        ambiguities = disambiguator.analyze_command_ambiguity(command)
        print(f"   Ambiguities detected: {len(ambiguities)}")
        
        for ambiguity in ambiguities:
            print(f"     - {ambiguity.ambiguity_type.value}: '{ambiguity.problematic_text}'")
            print(f"       Confidence: {ambiguity.confidence:.2f}")
            if ambiguity.possible_interpretations:
                print(f"       Interpretations: {ambiguity.possible_interpretations}")
        
        # Generate disambiguation response
        response = disambiguator.generate_disambiguation_response(command)
        
        print(f"   Primary question: {response.primary_question}")
        print(f"   Can proceed: {response.can_proceed_with_assumptions}")
        
        if response.alternative_suggestions:
            print(f"   Suggestions: {response.alternative_suggestions[:2]}")
        
        if response.assumptions_made:
            print(f"   Assumptions: {response.assumptions_made[0]}")
    
    # Test with context
    print(f"\n🧠 Testing context-aware disambiguation...")
    
    # Add some context
    context.add_conversation_turn("create a red ball", "Created red ball")
    context.add_conversation_turn("add a blue box", "Added blue box")
    
    # Test pronoun resolution
    contextual_commands = [
        "move it to the right",
        "place them together",
        "what will happen to it?"
    ]
    
    for command in contextual_commands:
        response = disambiguator.generate_disambiguation_response(command)
        print(f"   '{command}' → {len(response.ambiguities_detected)} ambiguities")
        if response.ambiguities_detected:
            print(f"     Question: {response.primary_question}")
    
    print(f"\n✅ Command disambiguation test completed!")
    print(f"🎯 Key capabilities demonstrated:")
    print(f"   • Multiple ambiguity type detection")
    print(f"   • Context-aware pronoun resolution")
    print(f"   • Intelligent clarifying questions")
    print(f"   • Alternative suggestion generation")
    print(f"   • Assumption-based proceeding")


if __name__ == "__main__":
    test_command_disambiguation()
