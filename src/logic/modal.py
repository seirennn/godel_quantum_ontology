"""
Modal logic implementation for the Divine Algorithm project.
"""

from enum import Enum
from typing import Dict, Set, List, Optional, Any

class ModalOperator(Enum):
    """Modal logic operators."""
    NECESSARY = 1
    POSSIBLE = 2
    CONTINGENT = 3
    IMPOSSIBLE = 4

class World:
    """Represents a possible world in modal logic."""
    
    def __init__(self, id: str, properties: Set[str], accessible_worlds: Optional[Set[str]] = None):
        """Initialize world with properties and accessible worlds."""
        self.id = id
        self.properties = properties
        self.accessible_worlds: Set[str] = accessible_worlds if accessible_worlds is not None else set()
    
    def add_accessible(self, world_id: str):
        """Add accessible world."""
        self.accessible_worlds.add(world_id)
    
    def is_accessible(self, world_id: str) -> bool:
        """Check if world is accessible."""
        return world_id in self.accessible_worlds
    
    def has_property(self, prop: str) -> bool:
        """Check if world has property."""
        return prop in self.properties

class ModalFrame:
    """Modal logic frame with possible worlds."""
    
    def __init__(self):
        self.worlds: Dict[str, World] = {}
        self.actual_world: Optional[str] = None
    
    def add_world(self, id: str, properties: Set[str]):
        """Add world to frame."""
        if id in self.worlds:
            raise ValueError(f"World {id} already exists")
        self.worlds[id] = World(id, properties)
    
    def add_accessibility(self, from_id: str, to_id: str):
        """Add accessibility relation."""
        if from_id not in self.worlds or to_id not in self.worlds:
            raise ValueError("Both worlds must exist")
        self.worlds[from_id].add_accessible(to_id)
    
    def get_accessible_worlds(self, world_id: str) -> Set[str]:
        """Get accessible worlds from given world."""
        if world_id not in self.worlds:
            raise ValueError(f"World {world_id} does not exist")
        return self.worlds[world_id].accessible_worlds
    
    def set_actual_world(self, world_id: str):
        """Set the actual world."""
        if world_id not in self.worlds:
            raise ValueError(f"World {world_id} does not exist")
        self.actual_world = world_id

class ModalLogic:
    """Modal logic implementation."""
    
    def __init__(self, frame: Optional[ModalFrame] = None):
        """Initialize modal logic with optional frame."""
        self.frame = frame if frame is not None else ModalFrame()
    
    def add_world(self, id: str, properties: Set[str]):
        """Add world to modal frame."""
        self.frame.add_world(id, properties)
    
    def add_accessibility(self, from_id: str, to_id: str):
        """Add accessibility relation between worlds."""
        self.frame.add_accessibility(from_id, to_id)
    
    def evaluate_necessity(self, prop: str, world_id: str) -> bool:
        """
        Evaluate whether property is necessary in world.
        
        Args:
            prop: Property to evaluate
            world_id: World to evaluate in
            
        Returns:
            True if property is necessary
        """
        if world_id not in self.frame.worlds:
            raise ValueError(f"World {world_id} does not exist")
            
        world = self.frame.worlds[world_id]
        
        # For necessity, property must be true in current world
        if not world.has_property(prop):
            return False
            
        # If no accessible worlds, property is necessary if world has it
        if not world.accessible_worlds:
            return True
            
        # And must be true in all accessible worlds
        return all(self.frame.worlds[w].has_property(prop) for w in world.accessible_worlds)
    
    def evaluate_possibility(self, prop: str, world_id: str) -> bool:
        """
        Evaluate whether property is possible in world.
        
        Args:
            prop: Property to evaluate
            world_id: World to evaluate in
            
        Returns:
            True if property is possible
        """
        if world_id not in self.frame.worlds:
            raise ValueError(f"World {world_id} does not exist")
            
        def check_world_and_accessible(world_id: str, visited: Set[str]) -> bool:
            if world_id in visited:
                return False
            visited.add(world_id)
            
            world = self.frame.worlds[world_id]
            if world.has_property(prop):
                return True
                
            for next_id in world.accessible_worlds:
                if check_world_and_accessible(next_id, visited):
                    return True
            return False
        
        return check_world_and_accessible(world_id, set())
    
    def evaluate_contingency(self, prop: str, world_id: str) -> bool:
        """
        Evaluate whether property is contingent in world.
        
        Args:
            prop: Property to evaluate
            world_id: World to evaluate in
            
        Returns:
            True if property is contingent
        """
        if world_id not in self.frame.worlds:
            raise ValueError(f"World {world_id} does not exist")
            
        return (self.evaluate_possibility(prop, world_id) and 
                not self.evaluate_necessity(prop, world_id))
    
    def determine_modality(self, prop: str, world_id: str) -> ModalOperator:
        """
        Determine modal operator for property in world.
        
        Args:
            prop: Property to evaluate
            world_id: World to evaluate in
            
        Returns:
            Modal operator for property
        """
        if world_id not in self.frame.worlds:
            raise ValueError(f"World {world_id} does not exist")
            
        world = self.frame.worlds[world_id]
        
        # If property is true in current world
        if world.has_property(prop):
            # If no accessible worlds, it's necessary
            if not world.accessible_worlds:
                return ModalOperator.NECESSARY
            # Check if any accessible world has the property
            has_accessible_with_prop = False
            for w in world.accessible_worlds:
                if self.frame.worlds[w].has_property(prop):
                    has_accessible_with_prop = True
                    break
            # If no accessible world has the property, but current world does, it's necessary
            if not has_accessible_with_prop:
                return ModalOperator.NECESSARY
            # If some accessible worlds have the property, it's necessary if all have it
            if all(self.frame.worlds[w].has_property(prop) for w in world.accessible_worlds):
                return ModalOperator.NECESSARY
            # Otherwise it's contingent
            return ModalOperator.CONTINGENT
        
        # If property is false in current world
        # Check if it's possible in any accessible world
        for w in world.accessible_worlds:
            if self.frame.worlds[w].has_property(prop):
                return ModalOperator.POSSIBLE
        return ModalOperator.IMPOSSIBLE
            
    def get_modal_operator(self, prop: str, world_id: str) -> ModalOperator:
        """Alias for determine_modality."""
        return self.determine_modality(prop, world_id)

def analyze_quantum_necessity(measurement_results: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze quantum measurement results using modal logic.
    
    Args:
        measurement_results: Dictionary of measurement results
        
    Returns:
        Dictionary of modal analysis results
    """
    logic = ModalLogic()
    
    # Create initial world
    logic.add_world("w0", {"quantum_state"})
    
    # Create worlds for each measurement outcome
    necessary_states = []
    possible_states = []
    contingent_states = []
    impossible_states = []
    
    for state, prob in measurement_results.items():
        if isinstance(prob, (int, float)):
            if prob > 0.99:  # High probability indicates necessity
                logic.add_world(state, {"measured"})
                logic.add_accessibility("w0", state)
                necessary_states.append(state)
                possible_states.append(state)
            elif prob > 0.1:  # Medium probability indicates contingency
                logic.add_world(state, {"measured"})
                logic.add_accessibility("w0", state)
                contingent_states.append(state)
                possible_states.append(state)
            elif prob > 0:  # Low probability indicates possibility
                logic.add_world(state, {"measured"})
                logic.add_accessibility("w0", state)
                possible_states.append(state)
            else:  # Zero probability indicates impossibility
                impossible_states.append(state)
    
    # Calculate ratios
    total_states = len(necessary_states) + len(contingent_states) + len(impossible_states)
    necessity_ratio = len(necessary_states) / total_states if total_states > 0 else 0.0
    contingency_ratio = len(contingent_states) / total_states if total_states > 0 else 0.0
    
    # Analyze necessity
    analysis = {
        "necessary": logic.evaluate_necessity("measured", "w0"),
        "possible": logic.evaluate_possibility("measured", "w0"),
        "modality": logic.determine_modality("measured", "w0"),
        "necessary_states": necessary_states,
        "possible_states": possible_states,
        "contingent_states": contingent_states,
        "impossible_states": impossible_states,
        "necessity_ratio": necessity_ratio,
        "contingency_ratio": contingency_ratio
    }
    
    # Override modality based on state ratios and accessibility
    if len(possible_states) == 0:
        analysis["modality"] = ModalOperator.IMPOSSIBLE
    elif len(necessary_states) > 0 and necessity_ratio > 0.5:
        analysis["modality"] = ModalOperator.NECESSARY
    elif len(contingent_states) > 0:
        analysis["modality"] = ModalOperator.CONTINGENT
    elif len(possible_states) > 0:
        analysis["modality"] = ModalOperator.POSSIBLE
    else:
        analysis["modality"] = ModalOperator.IMPOSSIBLE
    
    return analysis
