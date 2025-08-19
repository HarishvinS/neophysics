"""
Natural Conversation Interface
Enables natural conversation about physics with questions, explanations, and interactive discussions.
Goes beyond command execution to provide educational and exploratory physics conversations.
"""

import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from conversational_context import ConversationalContext, ContextType
from generative_understanding import ConceptualReasoner
from multi_step_command_parser import MultiStepCommandParser
from improved_physics_reasoning import ImprovedPhysicsReasoner
from dynamic_scene_representation import DynamicPhysicsScene


class ConversationMode(Enum):
    """Different modes of conversation."""
    COMMAND_EXECUTION = "command_execution"
    PHYSICS_EXPLANATION = "physics_explanation"
    EXPLORATORY_DISCUSSION = "exploratory_discussion"
    EDUCATIONAL_TUTORIAL = "educational_tutorial"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_BRAINSTORMING = "creative_brainstorming"


class ResponseType(Enum):
    """Types of responses the system can generate."""
    DIRECT_ANSWER = "direct_answer"
    EXPLANATION = "explanation"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    CLARIFICATION = "clarification"
    ENCOURAGEMENT = "encouragement"
    DEMONSTRATION = "demonstration"


@dataclass
class ConversationResponse:
    """Represents a response in the conversation."""
    response_id: str
    response_type: ResponseType
    content: str
    confidence: float
    follow_up_suggestions: List[str]
    educational_notes: List[str]
    demonstration_available: bool
    
    def to_dict(self):
        return {
            'response_id': self.response_id,
            'response_type': self.response_type.value,
            'content': self.content,
            'confidence': self.confidence,
            'follow_up_suggestions': self.follow_up_suggestions,
            'educational_notes': self.educational_notes,
            'demonstration_available': self.demonstration_available
        }


class NaturalConversationInterface:
    """Manages natural conversation about physics concepts and simulations."""
    
    def __init__(self):
        """Initialize natural conversation interface."""
        self.context = ConversationalContext()
        self.conceptual_reasoner = ConceptualReasoner()
        self.command_parser = MultiStepCommandParser(self.context)
        self.physics_reasoner = ImprovedPhysicsReasoner()
        
        # Conversation state
        self.current_mode = ConversationMode.COMMAND_EXECUTION
        self.current_scene = DynamicPhysicsScene("conversation_scene")
        self.conversation_personality = "friendly_educator"
        
        # Knowledge base for physics explanations
        self.physics_knowledge = self._build_physics_knowledge()
        self.conversation_templates = self._build_conversation_templates()
    
    def _build_physics_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Build knowledge base for physics explanations."""
        return {
            'gravity': {
                'simple_explanation': "Gravity pulls objects downward with a force of 9.81 m/s²",
                'detailed_explanation': "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, this creates a downward acceleration of 9.81 meters per second squared.",
                'examples': ["A ball falling", "Objects rolling down ramps", "Pendulum motion"],
                'related_concepts': ['acceleration', 'force', 'mass', 'weight']
            },
            'momentum': {
                'simple_explanation': "Momentum is mass times velocity - heavier or faster objects have more momentum",
                'detailed_explanation': "Momentum (p = mv) is conserved in collisions. When objects collide, the total momentum before equals the total momentum after.",
                'examples': ["Billiard ball collisions", "Car crashes", "Domino effects"],
                'related_concepts': ['velocity', 'mass', 'collision', 'conservation']
            },
            'friction': {
                'simple_explanation': "Friction opposes motion between surfaces in contact",
                'detailed_explanation': "Friction depends on the materials in contact and the normal force pressing them together. It converts kinetic energy into heat.",
                'examples': ["Objects sliding to a stop", "Rolling resistance", "Braking"],
                'related_concepts': ['energy', 'heat', 'surfaces', 'materials']
            },
            'energy': {
                'simple_explanation': "Energy can be kinetic (motion) or potential (stored) and is conserved",
                'detailed_explanation': "Energy cannot be created or destroyed, only converted between forms. Potential energy becomes kinetic energy and vice versa.",
                'examples': ["Ball on a hill", "Pendulum swinging", "Spring compression"],
                'related_concepts': ['kinetic', 'potential', 'conservation', 'work']
            },
            'collisions': {
                'simple_explanation': "When objects hit, they exchange momentum and energy",
                'detailed_explanation': "Elastic collisions conserve both momentum and kinetic energy. Inelastic collisions conserve momentum but lose kinetic energy.",
                'examples': ["Bouncing balls", "Car crashes", "Pool shots"],
                'related_concepts': ['momentum', 'energy', 'elastic', 'inelastic']
            }
        }
    
    def _build_conversation_templates(self) -> Dict[str, List[str]]:
        """Build templates for different types of responses."""
        return {
            'encouragement': [
                "Great question! Let me explain that.",
                "That's an interesting physics problem!",
                "I love exploring physics concepts like this!",
                "Excellent observation! Here's what's happening:"
            ],
            'explanation_intro': [
                "Here's how the physics works:",
                "Let me break this down for you:",
                "The key physics principle here is:",
                "From a physics perspective:"
            ],
            'suggestion': [
                "You might also want to try:",
                "Here's another interesting scenario:",
                "What if we also:",
                "Consider this variation:"
            ],
            'clarification': [
                "Could you clarify what you mean by:",
                "I want to make sure I understand - are you asking about:",
                "Just to be clear, do you want to:",
                "Help me understand better:"
            ],
            'demonstration_offer': [
                "Would you like me to demonstrate this?",
                "I can show you this in action!",
                "Want to see a simulation of this?",
                "Let me create a demo for you!"
            ]
        }
    
    def process_conversation_input(self, user_input: str) -> ConversationResponse:
        """Process user input and generate appropriate conversational response."""
        # Determine conversation mode and intent
        mode = self._determine_conversation_mode(user_input)
        self.current_mode = mode
        
        # Generate response based on mode
        if mode == ConversationMode.COMMAND_EXECUTION:
            response = self._handle_command_execution(user_input)
        elif mode == ConversationMode.PHYSICS_EXPLANATION:
            response = self._handle_physics_explanation(user_input)
        elif mode == ConversationMode.EXPLORATORY_DISCUSSION:
            response = self._handle_exploratory_discussion(user_input)
        elif mode == ConversationMode.EDUCATIONAL_TUTORIAL:
            response = self._handle_educational_tutorial(user_input)
        elif mode == ConversationMode.PROBLEM_SOLVING:
            response = self._handle_problem_solving(user_input)
        else:
            response = self._handle_creative_brainstorming(user_input)
        
        # Add to conversation context
        self.context.add_conversation_turn(
            user_input, response.content, self.current_scene, 
            response.educational_notes
        )
        
        return response
    
    def _determine_conversation_mode(self, user_input: str) -> ConversationMode:
        """Determine the appropriate conversation mode for the input."""
        input_lower = user_input.lower()

        # Data-driven approach for determining mode
        mode_keywords = {
            ConversationMode.COMMAND_EXECUTION: ['create', 'make', 'add', 'place', 'put', 'build', 'simulate', 'run'],
            ConversationMode.PHYSICS_EXPLANATION: ['what', 'how', 'why', 'explain', 'happens', 'works'],
            ConversationMode.EDUCATIONAL_TUTORIAL: ['teach', 'learn', 'understand', 'tutorial', 'lesson'],
            ConversationMode.PROBLEM_SOLVING: ['problem', 'solve', 'figure out', 'calculate', 'find'],
            ConversationMode.CREATIVE_BRAINSTORMING: ['idea', 'creative', 'interesting', 'cool', 'fun', 'experiment']
        }

        # Prioritize modes. Command execution is a strong indicator.
        mode_priority = [
            ConversationMode.COMMAND_EXECUTION,
            ConversationMode.PROBLEM_SOLVING,
            ConversationMode.EDUCATIONAL_TUTORIAL,
            ConversationMode.PHYSICS_EXPLANATION,
            ConversationMode.CREATIVE_BRAINSTORMING
        ]

        for mode in mode_priority:
            if mode in mode_keywords:
                if any(word in input_lower for word in mode_keywords[mode]):
                    return mode

        # Default fallback mode
        return ConversationMode.EXPLORATORY_DISCUSSION
    
    def _handle_command_execution(self, user_input: str) -> ConversationResponse:
        """Handle command execution with conversational feedback."""
        # Parse command sequence
        sequence = self.command_parser.parse_command_sequence(user_input)
        
        # Generate conversational response
        if sequence.overall_confidence > 0.8:
            content = f"Perfect! I'll {sequence.steps[0].parsed_action}"
            if len(sequence.steps) > 1:
                content += f" and then {len(sequence.steps) - 1} more steps"
            content += f". This should take about {sequence.estimated_duration:.1f} seconds."
        elif sequence.overall_confidence > 0.6:
            content = f"I think I understand - you want me to {sequence.steps[0].parsed_action}"
            if len(sequence.steps) > 1:
                content += f" followed by {len(sequence.steps) - 1} more actions"
            content += ". Let me know if that's not quite right!"
        else:
            content = f"I'm trying to understand your request. It seems like you want to {sequence.steps[0].parsed_action}. Could you clarify the details?"
        
        # Generate follow-up suggestions
        suggestions = [
            "Want to see what happens next?",
            "Should I predict the physics outcome?",
            "Any modifications needed?"
        ]
        
        # Educational notes
        educational_notes = []
        if any(step.command_type.value == 'create' for step in sequence.steps):
            educational_notes.append("Creating objects allows us to explore physics interactions")
        
        return ConversationResponse(
            response_id=f"cmd_resp_{int(time.time())}",
            response_type=ResponseType.DIRECT_ANSWER,
            content=content,
            confidence=sequence.overall_confidence,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )
    
    def _handle_physics_explanation(self, user_input: str) -> ConversationResponse:
        """Handle physics explanation requests."""
        input_lower = user_input.lower()
        
        # Identify physics concepts mentioned
        mentioned_concepts = []
        for concept in self.physics_knowledge.keys():
            if concept in input_lower:
                mentioned_concepts.append(concept)
        
        if mentioned_concepts:
            # Explain the primary concept
            primary_concept = mentioned_concepts[0]
            knowledge = self.physics_knowledge[primary_concept]
            
            # Choose explanation level based on question complexity
            if any(word in input_lower for word in ['simple', 'basic', 'quick']):
                explanation = knowledge['simple_explanation']
            else:
                explanation = knowledge['detailed_explanation']
            
            content = f"{random.choice(self.conversation_templates['explanation_intro'])} {explanation}"
            
            # Add examples if helpful
            if 'example' in input_lower or 'show' in input_lower:
                examples = knowledge['examples'][:2]
                content += f" For example: {', '.join(examples)}."
            
            # Generate educational notes
            educational_notes = [
                f"Key concept: {primary_concept}",
                f"Related topics: {', '.join(knowledge['related_concepts'][:3])}"
            ]
            
            # Suggest demonstrations
            suggestions = [
                f"Want to see a {primary_concept} demonstration?",
                "Should I create a scenario to show this?",
                f"Interested in exploring {knowledge['related_concepts'][0]}?"
            ]
            
            confidence = 0.9
        else:
            # General physics discussion
            content = "That's a great physics question! Could you be more specific about what aspect you'd like me to explain?"
            educational_notes = ["Physics covers many fascinating topics!"]
            suggestions = [
                "Ask about gravity, momentum, energy, or friction",
                "Want to explore a specific scenario?",
                "Should I suggest some interesting physics demos?"
            ]
            confidence = 0.6
        
        return ConversationResponse(
            response_id=f"phys_resp_{int(time.time())}",
            response_type=ResponseType.EXPLANATION,
            content=content,
            confidence=confidence,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )
    
    def _handle_exploratory_discussion(self, user_input: str) -> ConversationResponse:
        """Handle open-ended exploratory discussions."""
        content = f"{random.choice(self.conversation_templates['encouragement'])} "
        
        # Analyze what the user might be interested in
        if 'interesting' in user_input.lower():
            content += "There are so many fascinating physics phenomena! What catches your attention most?"
        elif 'fun' in user_input.lower():
            content += "Physics can be really fun! Let's explore something exciting together."
        else:
            content += "I'd love to explore physics concepts with you. What would you like to discover?"
        
        suggestions = [
            "Try creating some objects and see what happens",
            "Ask me about any physics concept",
            "Want to see some cool physics demonstrations?",
            "Let's build an interesting scenario together"
        ]
        
        educational_notes = [
            "Exploration is a great way to learn physics!",
            "Every physics interaction teaches us something new"
        ]
        
        return ConversationResponse(
            response_id=f"explore_resp_{int(time.time())}",
            response_type=ResponseType.QUESTION,
            content=content,
            confidence=0.8,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )
    
    def _handle_educational_tutorial(self, user_input: str) -> ConversationResponse:
        """Handle educational tutorial requests."""
        content = "I'd love to teach you about physics! "
        
        # Determine what they want to learn
        if 'basic' in user_input.lower() or 'beginner' in user_input.lower():
            content += "Let's start with fundamental concepts like gravity, momentum, and energy. Which interests you most?"
        elif 'advanced' in user_input.lower():
            content += "Great! We can explore complex topics like conservation laws, rotational dynamics, or wave mechanics."
        else:
            content += "What physics topic would you like to explore? I can explain concepts and create demonstrations."
        
        suggestions = [
            "Start with gravity and falling objects",
            "Learn about collisions and momentum",
            "Explore energy conservation",
            "Understand friction and motion"
        ]
        
        educational_notes = [
            "Learning physics is best done through hands-on exploration",
            "Each concept builds on previous understanding",
            "Demonstrations help visualize abstract concepts"
        ]
        
        return ConversationResponse(
            response_id=f"tutorial_resp_{int(time.time())}",
            response_type=ResponseType.EXPLANATION,
            content=content,
            confidence=0.9,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )
    
    def _handle_problem_solving(self, user_input: str) -> ConversationResponse:
        """Handle physics problem solving."""
        content = "Let's solve this physics problem together! "
        
        # Try to identify the type of problem
        if 'collision' in user_input.lower():
            content += "For collision problems, we need to consider momentum conservation and energy."
        elif 'motion' in user_input.lower():
            content += "For motion problems, we'll use kinematic equations and force analysis."
        elif 'energy' in user_input.lower():
            content += "For energy problems, we'll apply conservation of energy principles."
        else:
            content += "Could you describe the specific physics scenario you'd like to analyze?"
        
        suggestions = [
            "Describe the objects and their properties",
            "Tell me the initial conditions",
            "What do you want to find or predict?",
            "Should I create a visual demonstration?"
        ]
        
        educational_notes = [
            "Problem solving requires identifying relevant physics principles",
            "Breaking complex problems into steps helps",
            "Visualization often clarifies the physics"
        ]
        
        return ConversationResponse(
            response_id=f"problem_resp_{int(time.time())}",
            response_type=ResponseType.QUESTION,
            content=content,
            confidence=0.8,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )
    
    def _handle_creative_brainstorming(self, user_input: str) -> ConversationResponse:
        """Handle creative brainstorming about physics scenarios."""
        content = "I love creative physics exploration! "
        
        # Generate creative suggestions
        creative_ideas = [
            "What if we created a Rube Goldberg machine?",
            "How about exploring unusual object shapes?",
            "Want to experiment with different materials?",
            "Let's try some impossible physics scenarios!",
            "What about creating artistic physics patterns?"
        ]
        
        content += random.choice(creative_ideas)
        
        suggestions = [
            "Design a complex chain reaction",
            "Experiment with novel object shapes",
            "Create artistic physics patterns",
            "Build impossible scenarios for fun"
        ]
        
        educational_notes = [
            "Creativity enhances physics understanding",
            "Unusual scenarios often reveal interesting physics",
            "Experimentation leads to discovery"
        ]
        
        return ConversationResponse(
            response_id=f"creative_resp_{int(time.time())}",
            response_type=ResponseType.SUGGESTION,
            content=content,
            confidence=0.8,
            follow_up_suggestions=suggestions,
            educational_notes=educational_notes,
            demonstration_available=True
        )


def test_natural_conversation_interface():
    """Test the natural conversation interface."""
    print("Testing Natural Conversation Interface...")
    
    interface = NaturalConversationInterface()
    
    # Test different types of conversational inputs
    test_inputs = [
        "create a ball and place it on a ramp",
        "what happens when objects collide?",
        "explain gravity to me",
        "I want to learn about physics",
        "help me solve this collision problem",
        "let's try something creative and fun",
        "that's interesting, tell me more",
        "how does momentum work?",
        "can you teach me about energy?"
    ]
    
    print(f"✅ Testing {len(test_inputs)} conversational inputs...")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n🗣️ Input {i}: '{user_input}'")
        
        response = interface.process_conversation_input(user_input)
        
        print(f"   Mode: {interface.current_mode.value}")
        print(f"   Response type: {response.response_type.value}")
        print(f"   Content: {response.content}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Suggestions: {response.follow_up_suggestions[:2]}")
        
        if response.educational_notes:
            print(f"   Educational notes: {response.educational_notes[0]}")
        
        if response.demonstration_available:
            print(f"   📺 Demonstration available")
    
    # Test context awareness
    print(f"\n🧠 Testing context awareness...")
    
    # Simulate a conversation sequence
    conversation_sequence = [
        "create a red ball",
        "place it on a ramp", 
        "what will happen to it?",
        "why does that happen?"
    ]
    
    for user_input in conversation_sequence:
        response = interface.process_conversation_input(user_input)
        print(f"   '{user_input}' → {response.response_type.value}")
    
    # Test context summary
    context_summary = interface.context.get_context_summary()
    print(f"   Context: {context_summary['total_turns']} turns, {context_summary['active_entities']} entities")
    
    print(f"\n✅ Natural conversation interface test completed!")
    print(f"🎯 Key capabilities demonstrated:")
    print(f"   • Multi-mode conversation handling")
    print(f"   • Physics education and explanation")
    print(f"   • Context-aware responses")
    print(f"   • Educational note generation")
    print(f"   • Follow-up suggestion system")


if __name__ == "__main__":
    test_natural_conversation_interface()
