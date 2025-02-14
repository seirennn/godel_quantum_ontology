"""
Quantum measurement and analysis for the Divine Algorithm.
This module handles the measurement of quantum states and analysis
of results to determine the necessity of a non-contingent being.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.quantum_info import state_fidelity, Statevector

class QuantumMeasurement:
    """Handler for quantum measurements and analysis."""
    
    def __init__(self, shots: int = 1000):
        """
        Initialize measurement configuration.
        
        Args:
            shots: Number of measurement shots to perform
        """
        if shots <= 0:
            raise ValueError("Number of shots must be positive")
            
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        self.sampler = Sampler()
        
    def measure_system_stability(
        self,
        circuit: QuantumCircuit
    ) -> Dict[str, float]:
        """
        Measure the stability of the quantum system.
        
        Args:
            circuit: Quantum circuit to measure
            
        Returns:
            Dictionary containing stability metrics
        """
        if circuit is None:
            raise ValueError("Circuit cannot be None")
            
        # Create a new circuit with proper measurement setup
        qr = QuantumRegister(circuit.num_qubits, 'q')
        cr = ClassicalRegister(circuit.num_qubits, 'c')
        measured_circuit = QuantumCircuit(qr, cr)
        
        # Copy the original circuit's operations
        for instruction in circuit.data:
            if instruction[0].name != 'measure':  # Skip any existing measurements
                measured_circuit.append(instruction[0], instruction[1], instruction[2])
        
        # Add measurements for all qubits
        for i in range(circuit.num_qubits):
            measured_circuit.measure(qr[i], cr[i])
        
        # Execute circuit using Sampler primitive
        job = self.sampler.run([measured_circuit], shots=self.shots)
        result = job.result()
        counts = result.quasi_dists[0]
        
        # Convert quasi-distribution to counts format
        counts_dict = {format(int(state), 'b').zfill(circuit.num_qubits): int(count * self.shots)
                      for state, count in counts.items()}
        
        # Analyze results
        stability_metrics = {
            'quantum_entropy': self._calculate_entropy(counts_dict),
            'measurement_stability': self._analyze_measurement_stability(counts_dict),
            'state_coherence': self._estimate_coherence(counts_dict),
            'causal_strength': self._analyze_causal_strength(counts_dict)
        }
        
        # Normalize metrics to [0, 1]
        for key, value in stability_metrics.items():
            if not (0 <= value <= 1):
                stability_metrics[key] = 1.0 / (1.0 + abs(value))
        
        return stability_metrics
    
    def _calculate_entropy(self, counts: Dict[str, int]) -> float:
        """Calculate quantum entropy from measurement results."""
        total_shots = sum(counts.values())
        
        # Calculate Shannon entropy of measurements
        probabilities = [count/total_shots for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_measurement_stability(self, counts: Dict[str, int]) -> float:
        """Analyze the stability of measurements over multiple shots."""
        total_shots = sum(counts.values())
        
        # Calculate variance in measurement probabilities
        mean_prob = 1.0 / len(counts)
        actual_probs = [count/total_shots for count in counts.values()]
        variance = np.var(actual_probs)
        
        # Normalize stability score (0 = unstable, 1 = stable)
        max_variance = mean_prob * (1 - mean_prob)  # Maximum possible variance
        stability = 1.0 - (variance / max_variance if max_variance > 0 else 0.0)
        
        return max(0.0, min(1.0, stability))
    
    def _estimate_coherence(self, counts: Dict[str, int]) -> float:
        """Estimate quantum coherence from measurement results."""
        total_shots = sum(counts.values())
        
        # Look for measurement patterns indicating coherence
        coherent_counts = sum(
            count for state, count in counts.items()
            if self._is_coherent_state(state)
        )
        
        return coherent_counts / total_shots
    
    def _is_coherent_state(self, state: str) -> bool:
        """
        Check if a measured state indicates quantum coherence.
        
        Args:
            state: Binary string representing measured state
            
        Returns:
            Boolean indicating if state shows coherence
        """
        # Look for patterns indicating quantum superposition
        ones_count = state.count('1')
        zeros_count = state.count('0')
        
        # Check for balanced superposition
        if abs(ones_count - zeros_count) <= 1:
            return True
            
        # Check for interesting patterns
        for i in range(len(state) - 1):
            if state[i] == state[i + 1]:
                return False
                
        return True
    
    def _analyze_causal_strength(self, counts: Dict[str, int]) -> float:
        """Analyze the strength of causal relationships in measurements."""
        total_shots = sum(counts.values())
        
        # Look for patterns indicating strong causal relationships
        causal_patterns = 0
        for state, count in counts.items():
            # Check for sequential dependencies in state
            for i in range(len(state) - 1):
                if state[i] == '1' and state[i + 1] == '1':
                    causal_patterns += count
                    break
        
        return causal_patterns / total_shots

    def analyze_necessity_indicators(
        self,
        stability_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze metrics to determine indicators of necessary being.
        
        Args:
            stability_metrics: Dictionary of stability measurements
            
        Returns:
            Dictionary containing necessity analysis
        """
        # Define thresholds for necessity indicators
        ENTROPY_THRESHOLD = 0.7
        STABILITY_THRESHOLD = 0.6
        COHERENCE_THRESHOLD = 0.4
        CAUSALITY_THRESHOLD = 0.5
        
        # Analyze each metric against thresholds
        analysis = {
            'high_entropy': stability_metrics['quantum_entropy'] > ENTROPY_THRESHOLD,
            'low_stability': stability_metrics['measurement_stability'] < STABILITY_THRESHOLD,
            'quantum_coherence': stability_metrics['state_coherence'] > COHERENCE_THRESHOLD,
            'strong_causality': stability_metrics['causal_strength'] > CAUSALITY_THRESHOLD
        }
        
        # Calculate overall necessity score
        necessity_indicators = sum(1 for indicator in analysis.values() if indicator)
        analysis['necessity_score'] = necessity_indicators / len(analysis)
        
        # Determine if results suggest necessary being
        analysis['suggests_necessary_being'] = analysis['necessity_score'] > 0.75
        
        return analysis

def measure_system_stability(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Convenience function to measure system stability.
    
    Args:
        circuit: Quantum circuit to measure
        
    Returns:
        Dictionary containing stability metrics and necessity analysis
    """
    if not any(inst[0].name == 'measure' for inst in circuit.data):
        raise ValueError("Circuit must include measurements")
        
    measurement = QuantumMeasurement()
    stability_metrics = measurement.measure_system_stability(circuit)
    necessity_analysis = measurement.analyze_necessity_indicators(stability_metrics)
    
    return {
        'stability_metrics': stability_metrics,
        'necessity_analysis': necessity_analysis
    }