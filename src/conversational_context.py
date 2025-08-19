"""
Conversational Context System
Implements context awareness that remembers previous commands and can handle 
follow-up questions and references. Enables natural conversation about physics.
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject
from scene_representation import ObjectType, MaterialType, Vector3


class ContextType(Enum):
    """Types of conversational context."""
    OBJECT_REFERENCE = "object_reference"
    SCENE_STATE = "scene_state"
    PHYSICS_PREDICTION = "physics_prediction"
    USER_INTENT = "user_intent"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_id: str
    timestamp: float
    user_input: str
    system_response: str
    context_type: ContextType
    entities_mentioned: List[str]
    scene_state: Optional[Dict[str, Any]]
    physics_predictions: List[str]
    user_intent: str
    
    def to_dict(self):
        return {
            'turn_id': self.turn_id,
            'timestamp': self.timestamp,
            'user_input': self.user_input,
            'system_response': self.system_response,
            'context_type': self.context_type.value,
            'entities_mentioned': self.entities_mentioned,
            'scene_state': self.scene_state,
            'physics_predictions': self.physics_predictions,
            'user_intent': self.user_intent
        }


@dataclass
class ContextualEntity:
    """Represents an entity (object, concept) in conversational context."""
    entity_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any]
    last_mentioned: float
    mention_count: int
    aliases: List[str]
    
    def to_dict(self):
        return asdict(self)


class ConversationalContext:
    """Manages conversational context and entity tracking."""
    
    def __init__(self, max_history: int = 50):
        """Initialize conversational context."""
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.entities = {}  # entity_id -> ContextualEntity
        self.current_scene = None
        self.session_start = time.time()
        
        # Pronoun and reference resolution
        self.pronoun_mappings = {
            'it': None,
            'that': None,
            'this': None,
            'them': [],
            'those': [],
            'these': []
        }
        
        # Intent tracking
        self.current_intent = None
        self.pending_clarifications = []
    
    def add_conversation_turn(self, user_input: str, system_response: str, 
                            scene_state: Optional[DynamicPhysicsScene] = None,
                            physics_predictions: List[str] = None) -> ConversationTurn:
        """Add a new conversation turn to the context."""
        turn_id = f"turn_{len(self.conversation_history) + 1}_{int(time.time())}"
        
        # Extract entities and intent from user input
        entities_mentioned = self._extract_entities(user_input)
        user_intent = self._infer_intent(user_input)
        context_type = self._determine_context_type(user_input)
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=time.time(),
            user_input=user_input,
            system_response=system_response,
            context_type=context_type,
            entities_mentioned=entities_mentioned,
            scene_state=scene_state.to_dict() if scene_state else None,
            physics_predictions=physics_predictions or [],
            user_intent=user_intent
        )
        
        # Add to history
        self.conversation_history.append(turn)
        
        # Update entity tracking
        self._update_entities(entities_mentioned, turn)
        
        # Update pronoun references
        self._update_pronoun_references(entities_mentioned)
        
        # Update current scene
        if scene_state:
            self.current_scene = scene_state
        
        return turn
    
    def resolve_references(self, text: str) -> str:
        """Resolve pronouns and references in text to specific entities."""
        resolved_text = text
        
        # Handle pronouns
        for pronoun, entity in self.pronoun_mappings.items():
            if pronoun in text.lower():
                if entity and isinstance(entity, str):
                    resolved_text = resolved_text.replace(pronoun, entity)
                elif entity and isinstance(entity, list) and entity:
                    resolved_text = resolved_text.replace(pronoun, ', '.join(entity))
        
        # Handle "the [object]" references
        resolved_text = self._resolve_definite_references(resolved_text)
        
        return resolved_text
    
    def get_relevant_context(self, query: str, max_turns: int = 5) -> List[ConversationTurn]:
        """Get relevant conversation context for a query."""
        # Get recent turns
        recent_turns = list(self.conversation_history)[-max_turns:]
        
        # Filter for relevance
        relevant_turns = []
        query_words = set(query.lower().split())
        
        for turn in recent_turns:
            # Check if turn mentions similar entities or concepts
            turn_words = set(turn.user_input.lower().split())
            overlap = len(query_words.intersection(turn_words))
            
            if overlap > 0 or any(entity in query.lower() for entity in turn.entities_mentioned):
                relevant_turns.append(turn)
        
        return relevant_turns
    
    def get_current_entities(self) -> Dict[str, ContextualEntity]:
        """Get currently active entities in the conversation."""
        # Return entities mentioned in the last 10 minutes
        cutoff_time = time.time() - 600  # 10 minutes
        
        active_entities = {}
        for entity_id, entity in self.entities.items():
            if entity.last_mentioned > cutoff_time:
                active_entities[entity_id] = entity
        
        return active_entities
    
    def suggest_follow_ups(self) -> List[str]:
        """Suggest natural follow-up questions or actions."""
        suggestions = []
        
        if not self.conversation_history:
            return ["Try creating some objects to get started!"]
        
        last_turn = self.conversation_history[-1]
        
        # Suggest based on last intent
        if last_turn.user_intent == "create_object":
            suggestions.extend([
                "Would you like to add more objects?",
                "Should I run a physics simulation?",
                "Want to modify any properties?"
            ])
        
        elif last_turn.user_intent == "physics_question":
            suggestions.extend([
                "Would you like to see this in action?",
                "Should I create a demonstration?",
                "Want to try a different scenario?"
            ])
        
        elif last_turn.user_intent == "modify_scene":
            suggestions.extend([
                "How does this look now?",
                "Ready to run the simulation?",
                "Any other changes needed?"
            ])
        
        # Suggest based on current scene
        if self.current_scene and self.current_scene.get_object_count() > 0:
            suggestions.extend([
                "What do you think will happen?",
                "Should I predict the physics outcome?",
                "Want to add initial velocities?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity mentions from text."""
        entities = []
        
        # Common physics objects
        object_keywords = [
            'ball', 'sphere', 'box', 'cube', 'ramp', 'plane', 'cylinder',
            'pendulum', 'spring', 'chain', 'rope', 'bridge', 'car', 'wheel',
            'domino', 'tower', 'wall', 'platform', 'lever', 'pulley'
        ]
        
        # Materials
        material_keywords = [
            'wood', 'metal', 'steel', 'rubber', 'plastic', 'glass', 'stone',
            'aluminum', 'copper', 'iron', 'concrete', 'ceramic'
        ]
        
        # Properties
        property_keywords = [
            'heavy', 'light', 'large', 'small', 'fast', 'slow', 'red', 'blue',
            'green', 'yellow', 'black', 'white', 'curved', 'straight', 'hollow'
        ]
        
        text_lower = text.lower()
        
        # Extract object mentions
        for keyword in object_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        # Extract material mentions
        for keyword in material_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        # Extract property mentions
        for keyword in property_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        return list(set(entities))  # Remove duplicates
    
    def _infer_intent(self, text: str) -> str:
        """Infer user intent from text."""
        text_lower = text.lower()
        
        # Creation intents
        if any(word in text_lower for word in ['create', 'make', 'add', 'build', 'place', 'put']):
            return "create_object"
        
        # Modification intents
        if any(word in text_lower for word in ['move', 'change', 'modify', 'adjust', 'remove', 'delete']):
            return "modify_scene"
        
        # Physics question intents
        if any(word in text_lower for word in ['what', 'how', 'why', 'will', 'happen', 'predict']):
            return "physics_question"
        
        # Simulation intents
        if any(word in text_lower for word in ['run', 'simulate', 'start', 'play', 'execute']):
            return "run_simulation"
        
        # Clarification intents
        if any(word in text_lower for word in ['explain', 'clarify', 'mean', 'understand']):
            return "clarification"
        
        return "general"
    
    def _determine_context_type(self, text: str) -> ContextType:
        """Determine the type of context for this input."""
        text_lower = text.lower()
        
        # Check for object references
        if any(pronoun in text_lower for pronoun in ['it', 'that', 'this', 'them']):
            return ContextType.OBJECT_REFERENCE
        
        # Check for follow-up questions
        if any(word in text_lower for word in ['also', 'then', 'next', 'after']):
            return ContextType.FOLLOW_UP
        
        # Check for physics predictions
        if any(word in text_lower for word in ['what will', 'what happens', 'predict']):
            return ContextType.PHYSICS_PREDICTION
        
        # Check for clarifications
        if any(word in text_lower for word in ['what do you mean', 'clarify', 'explain']):
            return ContextType.CLARIFICATION
        
        return ContextType.USER_INTENT
    
    def _update_entities(self, entities_mentioned: List[str], turn: ConversationTurn):
        """Update entity tracking with new mentions."""
        for entity_name in entities_mentioned:
            entity_id = entity_name.lower().replace(' ', '_')
            
            if entity_id in self.entities:
                # Update existing entity
                entity = self.entities[entity_id]
                entity.last_mentioned = turn.timestamp
                entity.mention_count += 1
            else:
                # Create new entity
                entity = ContextualEntity(
                    entity_id=entity_id,
                    entity_type=self._classify_entity_type(entity_name),
                    name=entity_name,
                    properties={},
                    last_mentioned=turn.timestamp,
                    mention_count=1,
                    aliases=[entity_name]
                )
                self.entities[entity_id] = entity
    
    def _classify_entity_type(self, entity_name: str) -> str:
        """Classify the type of an entity."""
        entity_lower = entity_name.lower()
        
        if entity_lower in ['ball', 'sphere', 'box', 'cube', 'ramp', 'cylinder']:
            return 'physics_object'
        elif entity_lower in ['wood', 'metal', 'rubber', 'plastic', 'glass']:
            return 'material'
        elif entity_lower in ['heavy', 'light', 'large', 'small', 'red', 'blue']:
            return 'property'
        else:
            return 'unknown'
    
    def _update_pronoun_references(self, entities_mentioned: List[str]):
        """Update pronoun reference mappings."""
        if entities_mentioned:
            # Update singular pronouns to refer to the last mentioned entity
            last_entity = entities_mentioned[-1]
            self.pronoun_mappings['it'] = last_entity
            self.pronoun_mappings['that'] = last_entity
            self.pronoun_mappings['this'] = last_entity
            
            # Update plural pronouns to refer to all mentioned entities
            if len(entities_mentioned) > 1:
                self.pronoun_mappings['them'] = entities_mentioned
                self.pronoun_mappings['those'] = entities_mentioned
                self.pronoun_mappings['these'] = entities_mentioned
    
    def _resolve_definite_references(self, text: str) -> str:
        """Resolve definite references like 'the ball' to specific objects."""
        words = text.split()
        resolved_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word.lower() == 'the' and i + 1 < len(words):
                next_word = words[i + 1].lower()
                
                # Check if next word is a known entity
                if next_word in [entity.name.lower() for entity in self.entities.values()]:
                    # Find the most recently mentioned instance
                    matching_entities = [
                        entity for entity in self.entities.values()
                        if entity.name.lower() == next_word
                    ]
                    
                    if matching_entities:
                        most_recent = max(matching_entities, key=lambda e: e.last_mentioned)
                        resolved_words.append(f"the {most_recent.name}")
                        i += 2  # Skip both 'the' and the object name
                        continue
            
            resolved_words.append(word)
            i += 1
        
        return ' '.join(resolved_words)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversational context."""
        return {
            'session_duration': time.time() - self.session_start,
            'total_turns': len(self.conversation_history),
            'active_entities': len(self.get_current_entities()),
            'current_scene_objects': self.current_scene.get_object_count() if self.current_scene else 0,
            'recent_intents': [turn.user_intent for turn in list(self.conversation_history)[-5:]],
            'pending_clarifications': len(self.pending_clarifications)
        }


def test_conversational_context():
    """Test the conversational context system."""
    print("Testing Conversational Context System...")
    
    # Create context manager
    context = ConversationalContext()
    
    # Simulate conversation
    test_conversation = [
        ("create a red ball", "Created a red ball at position (0, 0, 1)"),
        ("place it on a ramp", "Placed the ball on the ramp"),
        ("what will happen?", "The ball will roll down the ramp due to gravity"),
        ("add another ball", "Added a second ball to the scene"),
        ("make them both blue", "Changed both balls to blue color")
    ]
    
    print(f"✅ Simulating conversation with {len(test_conversation)} turns...")
    
    for user_input, system_response in test_conversation:
        turn = context.add_conversation_turn(user_input, system_response)
        
        print(f"\nTurn: '{user_input}'")
        print(f"  Intent: {turn.user_intent}")
        print(f"  Context: {turn.context_type.value}")
        print(f"  Entities: {turn.entities_mentioned}")
        
        # Test reference resolution
        resolved = context.resolve_references(user_input)
        if resolved != user_input:
            print(f"  Resolved: '{resolved}'")
    
    # Test context retrieval
    relevant_context = context.get_relevant_context("what about the ball?")
    print(f"\n✅ Relevant context for 'what about the ball?': {len(relevant_context)} turns")
    
    # Test follow-up suggestions
    suggestions = context.suggest_follow_ups()
    print(f"✅ Follow-up suggestions: {suggestions}")
    
    # Test entity tracking
    entities = context.get_current_entities()
    print(f"✅ Active entities: {list(entities.keys())}")
    
    # Test context summary
    summary = context.get_context_summary()
    print(f"✅ Context summary: {summary}")
    
    print("✅ Conversational context test completed!")


if __name__ == "__main__":
    test_conversational_context()
