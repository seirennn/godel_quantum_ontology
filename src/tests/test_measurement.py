"""
Tests for quantum measurement operations.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from ..quantum.measurement import (
    QuantumMeasurement,
    measure_system_stability
)
from ..quantum.circuits import create_contingency_circuit

class TestQuantumMeasurement(unittest.TestCase):
    """Test cases for quantum measurement operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 3
        self.depth = 2
        self.shots = 1000
        self.measurement = QuantumMeasurement(shots=self.shots)
        self.sampler = Sampler()
    
    def test_measure_system_stability(self):
        """Test measurement of system stability."""
        # Create test circuit
        circuit = create_contingency_circuit(self.num_qubits, self.depth)
        
        # Measure stability
        stability_metrics = self.measurement.measure_system_stability(circuit)
        
        # Verify metrics
        self.assertIn('quantum_entropy', stability_metrics)
        self.assertIn('measurement_stability', stability_metrics)
        self.assertIn('state_coherence', stability_metrics)
        self.assertIn('causal_strength', stability_metrics)
        
        # Check metric ranges
        for metric in stability_metrics.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
    
    def test_entropy_calculation(self):
        """Test quantum entropy calculation."""
        # Create simple test circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)  # Create superposition
        circuit.cx(0, 1)  # Entangle qubits
        circuit.measure_all()
        
        # Calculate entropy
        job = self.sampler.run([circuit], shots=self.shots)
        result = job.result()
        counts = result.quasi_dists[0]
        counts_dict = {format(int(state), 'b').zfill(circuit.num_qubits): int(count * self.shots)
                      for state, count in counts.items()}
        entropy = self.measurement._calculate_entropy(counts_dict)
        
        # Verify entropy
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, np.log2(2**circuit.num_qubits))
    
    def test_measurement_stability(self):
        """Test measurement stability analysis."""
        # Create test circuit with proper classical bits
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)  # Equal number of quantum and classical bits
        circuit.h(range(self.num_qubits))  # Create superposition
        circuit.measure(range(self.num_qubits), range(self.num_qubits))  # Measure all qubits
        
        # Execute and analyze stability
        job = self.sampler.run([circuit], shots=self.shots)
        result = job.result()
        counts = result.quasi_dists[0]
        counts_dict = {format(int(state), 'b').zfill(circuit.num_qubits): int(count * self.shots)
                      for state, count in counts.items()}
        stability = self.measurement._analyze_measurement_stability(counts_dict)
        
        # Verify stability metric
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_coherence_estimation(self):
        """Test quantum coherence estimation."""
        # Create coherent state
        circuit = QuantumCircuit(1)
        circuit.h(0)  # Create superposition
        circuit.measure_all()
        
        # Estimate coherence
        job = self.sampler.run([circuit], shots=self.shots)
        result = job.result()
        counts = result.quasi_dists[0]
        counts_dict = {format(int(state), 'b').zfill(circuit.num_qubits): int(count * self.shots)
                      for state, count in counts.items()}
        coherence = self.measurement._estimate_coherence(counts_dict)
        
        # Verify coherence
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_causal_strength_analysis(self):
        """Test causal strength analysis."""
        # Create causally connected circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        # Analyze causal strength
        job = self.sampler.run([circuit], shots=self.shots)
        result = job.result()
        counts = result.quasi_dists[0]
        counts_dict = {format(int(state), 'b').zfill(circuit.num_qubits): int(count * self.shots)
                      for state, count in counts.items()}
        causal_strength = self.measurement._analyze_causal_strength(counts_dict)
        
        # Verify causal strength
        self.assertGreaterEqual(causal_strength, 0.0)
        self.assertLessEqual(causal_strength, 1.0)
    
    def test_necessity_indicators(self):
        """Test analysis of necessity indicators."""
        # Create test circuit
        circuit = create_contingency_circuit(self.num_qubits, self.depth)
        
        # Get stability metrics
        stability_metrics = self.measurement.measure_system_stability(circuit)
        
        # Analyze necessity indicators
        necessity_analysis = self.measurement.analyze_necessity_indicators(stability_metrics)
        
        # Verify analysis results
        self.assertIn('high_entropy', necessity_analysis)
        self.assertIn('low_stability', necessity_analysis)
        self.assertIn('quantum_coherence', necessity_analysis)
        self.assertIn('strong_causality', necessity_analysis)
        self.assertIn('necessity_score', necessity_analysis)
        self.assertIn('suggests_necessary_being', necessity_analysis)
        
        # Check score range
        self.assertGreaterEqual(necessity_analysis['necessity_score'], 0.0)
        self.assertLessEqual(necessity_analysis['necessity_score'], 1.0)
    
    def test_measurement_consistency(self):
        """Test consistency of measurements."""
        circuit = create_contingency_circuit(self.num_qubits, self.depth)
        
        # Perform multiple measurements
        results = []
        for _ in range(5):
            metrics = self.measurement.measure_system_stability(circuit)
            results.append(metrics)
        
        # Check consistency
        for metric_name in results[0].keys():
            values = [result[metric_name] for result in results]
            std_dev = np.std(values)
            # Ensure measurements are reasonably consistent
            self.assertLess(std_dev, 0.2)
    
    def test_invalid_measurements(self):
        """Test handling of invalid measurements."""
        # Test with invalid circuit
        with self.assertRaises(ValueError):
            self.measurement.measure_system_stability(None)
        
        # Test with circuit without measurements
        circuit = QuantumCircuit(1)
        circuit.h(0)  # Only superposition, no measurement
        with self.assertRaises(ValueError):
            measure_system_stability(circuit)
        
        # Test with invalid number of shots
        with self.assertRaises(ValueError):
            QuantumMeasurement(shots=0)

if __name__ == '__main__':
    unittest.main()