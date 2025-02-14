"""
Tests for wave function calculations.
"""

import unittest
import numpy as np
from ..physics.wave_function import (
    WaveFunction,
    ContingencyAnalyzer,
    analyze_wave_function
)

class TestWaveFunction(unittest.TestCase):
    """Test cases for wave function operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 2
        self.wave_function = WaveFunction(self.num_qubits)
    
    def test_initialization(self):
        """Test wave function initialization."""
        # Test default initialization to |0⟩ state
        self.assertEqual(self.wave_function.dimension, 2**self.num_qubits)
        self.assertTrue(np.allclose(self.wave_function.amplitudes[0], 1.0))
        self.assertTrue(np.allclose(self.wave_function.amplitudes[1:], 0.0))
    
    def test_set_state(self):
        """Test setting wave function state."""
        # Create test state vector
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.wave_function.set_state(state)
        
        # Verify state
        self.assertTrue(np.allclose(self.wave_function.amplitudes, state))
        
        # Test normalization
        unnormalized = np.array([2.0, 2.0, 0, 0])
        self.wave_function.set_state(unnormalized)
        self.assertTrue(np.allclose(
            np.sum(np.abs(self.wave_function.amplitudes)**2),
            1.0
        ))
    
    def test_apply_operator(self):
        """Test applying quantum operators."""
        # Create Hadamard-like operator
        operator = np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2)
        operator = np.kron(operator, operator)  # Extend to 2 qubits
        
        # Apply operator
        self.wave_function.apply_operator(operator)
        
        # Verify unitarity preservation
        self.assertTrue(np.allclose(
            np.sum(np.abs(self.wave_function.amplitudes)**2),
            1.0
        ))
    
    def test_get_probabilities(self):
        """Test probability calculations."""
        # Create superposition state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.wave_function.set_state(state)
        
        # Get probabilities
        probs = self.wave_function.get_probabilities()
        
        # Verify probabilities
        self.assertAlmostEqual(probs['00'], 0.5)
        self.assertAlmostEqual(probs['01'], 0.5)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
    
    def test_collapse_to_eigenstate(self):
        """Test wave function collapse."""
        # Create operator
        operator = np.diag([1.0, -1.0, -1.0, 1.0])
        
        # Perform collapse
        eigenvalue, new_wf = self.wave_function.collapse_to_eigenstate(operator)
        
        # Verify results
        self.assertIsInstance(eigenvalue, float)
        self.assertIsInstance(new_wf, WaveFunction)
        self.assertTrue(np.allclose(
            np.sum(np.abs(new_wf.amplitudes)**2),
            1.0
        ))
    
    def test_evolve(self):
        """Test wave function evolution."""
        # Create Hamiltonian
        hamiltonian = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Evolve state
        self.wave_function.evolve(hamiltonian, 1.0)
        
        # Verify unitarity preservation
        self.assertTrue(np.allclose(
            np.sum(np.abs(self.wave_function.amplitudes)**2),
            1.0
        ))
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        # Create operator
        operator = np.diag([1.0, -1.0, -1.0, 1.0])
        
        # Calculate expectation value
        expectation = self.wave_function.get_expectation_value(operator)
        
        # Verify result is real for Hermitian operator
        self.assertTrue(np.isreal(expectation))

class TestContingencyAnalyzer(unittest.TestCase):
    """Test cases for contingency analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.wave_function = WaveFunction(2)
        self.analyzer = ContingencyAnalyzer(self.wave_function)
    
    def test_analyze_contingency(self):
        """Test contingency analysis."""
        # Analyze contingency
        metrics = self.analyzer.analyze_contingency()
        
        # Verify metrics
        self.assertIn('superposition_measure', metrics)
        self.assertIn('entanglement_entropy', metrics)
        self.assertIn('state_complexity', metrics)
        self.assertIn('causal_dependency', metrics)
        
        # Check metric ranges
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
    
    def test_measure_superposition(self):
        """Test superposition measurement."""
        # Create superposition state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.wave_function.set_state(state)
        
        # Measure superposition
        superposition = self.analyzer._measure_superposition()
        
        # Verify measurement
        self.assertGreater(superposition, 0.0)
        self.assertLessEqual(superposition, 1.0)
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        # Create entangled state
        state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # Bell state
        self.wave_function.set_state(state)
        
        # Calculate entropy
        entropy = self.analyzer._calculate_entanglement_entropy()
        
        # Verify entropy
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, np.log2(2))
    
    def test_wave_function_analysis(self):
        """Test complete wave function analysis."""
        # Create test Hamiltonian
        hamiltonian = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Analyze wave function
        analysis = analyze_wave_function(self.wave_function, hamiltonian)
        
        # Verify analysis structure
        self.assertIn('superposition_measure', analysis)
        self.assertIn('entanglement_entropy', analysis)
        self.assertIn('state_complexity', analysis)
        self.assertIn('causal_dependency', analysis)
        self.assertIn('energy', analysis)
        self.assertIn('stability', analysis)

if __name__ == '__main__':
    unittest.main()