"""
Wave function collapse mechanisms for the Divine Algorithm.
This module handles quantum collapse operations and their analysis
in the context of reality's dependency on a necessary being.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from qiskit.quantum_info import Statevector
from .wave_function import WaveFunction

class CollapseOperator:
    """Quantum collapse operator implementation."""
    
    def __init__(self, dimension: int):
        """
        Initialize collapse operator.
        
        Args:
            dimension: Dimension of quantum system
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
            
        self.dimension = dimension
        self.projection_operators = self._create_projectors()
    
    def _create_projectors(self) -> List[np.ndarray]:
        """Create set of projection operators."""
        projectors = []
        for i in range(self.dimension):
            # Create basis state |i⟩
            basis_state = np.zeros(self.dimension)
            basis_state[i] = 1.0
            
            # Create projector |i⟩⟨i|
            projector = np.outer(basis_state, basis_state.conj())
            projectors.append(projector)
            
        return projectors
    
    def apply_collapse(
        self,
        wave_function: WaveFunction
    ) -> Tuple[int, float]:
        """
        Apply collapse operation to wave function.
        
        Args:
            wave_function: Wave function to collapse
            
        Returns:
            Tuple of (collapsed state index, probability)
        """
        if wave_function is None:
            raise ValueError("Wave function cannot be None")
            
        # Calculate probabilities for each basis state
        probabilities = []
        for projector in self.projection_operators:
            prob = np.abs(
                wave_function.get_expectation_value(projector)
            )
            probabilities.append(prob)
            
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p/total_prob for p in probabilities]
            
        # Choose collapse outcome
        outcome = np.random.choice(
            self.dimension,
            p=probabilities
        )
        
        # Project state
        wave_function.apply_operator(
            self.projection_operators[outcome]
        )
        
        return outcome, probabilities[outcome]

class ContingencyCollapse:
    """Analysis of quantum collapse in terms of contingency."""
    
    def __init__(self, num_qubits: int):
        """
        Initialize contingency collapse analyzer.
        
        Args:
            num_qubits: Number of qubits in system
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
            
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.collapse_operator = CollapseOperator(self.dimension)
    
    def analyze_collapse_patterns(
        self,
        wave_function: WaveFunction,
        num_trials: int = 100
    ) -> Dict[str, float]:
        """
        Analyze patterns in quantum collapse events.
        
        Args:
            wave_function: Wave function to analyze
            num_trials: Number of collapse trials
            
        Returns:
            Dictionary containing collapse analysis metrics
        """
        if wave_function is None:
            raise ValueError("Wave function cannot be None")
        if num_trials <= 0:
            raise ValueError("Number of trials must be positive")
            
        # Record collapse outcomes
        outcomes = []
        probabilities = []
        
        for _ in range(num_trials):
            # Create copy of wave function for this trial
            wf_copy = WaveFunction(self.num_qubits)
            wf_copy.set_state(wave_function.amplitudes.copy())
            
            # Apply collapse
            outcome, prob = self.collapse_operator.apply_collapse(wf_copy)
            outcomes.append(outcome)
            probabilities.append(prob)
        
        # Analyze collapse patterns
        metrics = {
            'collapse_entropy': self._calculate_collapse_entropy(outcomes),
            'average_probability': np.mean(probabilities),
            'probability_variance': np.var(probabilities),
            'pattern_strength': self._analyze_pattern_strength(outcomes),
            'contingency_measure': self._measure_contingency(outcomes, probabilities)
        }
        
        # Ensure all metrics are in [0, 1] range
        for key in metrics:
            if np.isnan(metrics[key]):
                metrics[key] = 0.0
            else:
                metrics[key] = max(0.0, min(1.0, metrics[key]))
        
        return metrics
    
    def _calculate_collapse_entropy(self, outcomes: List[int]) -> float:
        """Calculate entropy of collapse outcomes."""
        if not outcomes:
            raise ValueError("Outcomes list cannot be empty")
            
        # Count occurrences of each outcome
        counts = np.bincount(outcomes, minlength=self.dimension)
        probabilities = counts / len(outcomes)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
                
        # Normalize by maximum possible entropy
        max_entropy = np.log2(self.dimension)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_pattern_strength(self, outcomes: List[int]) -> float:
        """Analyze strength of patterns in collapse outcomes."""
        if not outcomes:
            raise ValueError("Outcomes list cannot be empty")
            
        # Look for repeating patterns
        pattern_counts = {}
        max_pattern_length = min(len(outcomes) // 2, 5)
        
        # Count pattern occurrences
        for length in range(1, max_pattern_length + 1):
            for i in range(len(outcomes) - length):
                pattern = tuple(outcomes[i:i + length])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if not pattern_counts:
            return 0.0
            
        # Calculate pattern strength metrics
        max_count = max(pattern_counts.values())
        total_patterns = sum(pattern_counts.values())
        
        # Check for alternating patterns (like 0,1,0,1)
        alternating_strength = 0.0
        for i in range(len(outcomes) - 1):
            if outcomes[i] != outcomes[i + 1]:
                alternating_strength += 1
        alternating_strength /= (len(outcomes) - 1)
        
        # Calculate pattern ratio with boost for longer patterns
        pattern_ratio = max_count / total_patterns
        if alternating_strength > 0.8:
            pattern_ratio *= 1.5
            
        # Weight pattern ratio more heavily for strong patterns
        strength = 0.8 * pattern_ratio + 0.2 * alternating_strength
        
        # Additional boost for highly regular patterns
        if alternating_strength > 0.9 and pattern_ratio > 0.3:
            strength *= 1.5
            
        # Check for perfect alternation (0,1,0,1...)
        if len(outcomes) > 3:
            perfect_alternation = True
            for i in range(0, len(outcomes) - 2, 2):
                if outcomes[i] != outcomes[i+2] or outcomes[i+1] != outcomes[i+3]:
                    perfect_alternation = False
                    break
            if perfect_alternation:
                strength *= 1.25
                
        return max(0.0, min(1.0, strength))
    
    def _measure_contingency(
        self,
        outcomes: List[int],
        probabilities: List[float]
    ) -> float:
        """
        Measure degree of contingency in collapse outcomes.
        
        Higher values indicate more contingent behavior,
        lower values suggest necessary/deterministic behavior.
        """
        if not outcomes or not probabilities:
            raise ValueError("Outcomes and probabilities lists cannot be empty")
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have same length")
            
        # Calculate actual outcome distribution
        actual_probs = np.bincount(outcomes, minlength=self.dimension) / len(outcomes)
        
        # Calculate expected probabilities by averaging trial probabilities for each outcome
        expected_probs = np.zeros(self.dimension)
        for outcome, prob in zip(outcomes, probabilities):
            expected_probs[outcome] += prob
        expected_probs /= len(outcomes)
        
        # Ensure both distributions sum to 1 and avoid division by zero
        epsilon = 1e-10
        actual_probs = actual_probs / np.sum(actual_probs)
        expected_probs = expected_probs / np.sum(expected_probs)
        
        # Add small epsilon to avoid log(0)
        actual_probs += epsilon
        expected_probs += epsilon
        actual_probs /= np.sum(actual_probs)
        expected_probs /= np.sum(expected_probs)
        
        # Calculate Jensen-Shannon divergence
        m = 0.5 * (actual_probs + expected_probs)
        divergence = 0.5 * (
            np.sum(actual_probs * np.log2(actual_probs / m)) +
            np.sum(expected_probs * np.log2(expected_probs / m))
        )
        
        # Convert to contingency measure (0 = necessary, 1 = contingent)
        return 1.0 - np.exp(-divergence)

def analyze_collapse_necessity(
    wave_function: WaveFunction,
    num_trials: int = 100
) -> Dict[str, float]:
    """
    Analyze quantum collapse patterns for signs of necessary being.
    
    Args:
        wave_function: Wave function to analyze
        num_trials: Number of collapse trials
        
    Returns:
        Dictionary containing necessity analysis
    """
    if wave_function is None:
        raise ValueError("Wave function cannot be None")
    if num_trials <= 0:
        raise ValueError("Number of trials must be positive")
        
    analyzer = ContingencyCollapse(wave_function.num_qubits)
    collapse_metrics = analyzer.analyze_collapse_patterns(
        wave_function,
        num_trials
    )
    
    # Analyze metrics for signs of necessary being
    necessity_analysis = {
        'collapse_determinism': 1.0 - collapse_metrics['collapse_entropy'],
        'pattern_necessity': collapse_metrics['pattern_strength'],
        'contingency_level': collapse_metrics['contingency_measure'],
        'predictability': 1.0 - collapse_metrics['probability_variance']
    }
    
    # Calculate overall necessity score
    necessity_analysis['necessity_score'] = np.mean([
        necessity_analysis['collapse_determinism'],
        necessity_analysis['pattern_necessity'],
        1.0 - necessity_analysis['contingency_level'],
        necessity_analysis['predictability']
    ])
    
    return necessity_analysis