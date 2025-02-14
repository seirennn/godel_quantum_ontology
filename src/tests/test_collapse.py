"""
Tests for wave function collapse mechanisms.
"""

import unittest
import numpy as np
from ..physics.collapse import (
    CollapseOperator,
    ContingencyCollapse,
    analyze_collapse_necessity
)
from ..physics.wave_function import WaveFunction

class TestCollapseOperator(unittest.TestCase):
    """Test cases for collapse operator operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 2
        self.dimension = 2**self.num_qubits
        self.collapse_operator = CollapseOperator(self.dimension)
        self.wave_function = WaveFunction(self.num_qubits)
    
    def test_projector_creation(self):
        """Test creation of projection operators."""
        projectors = self.collapse_operator._create_projectors()
        
        # Verify number of projectors
        self.assertEqual(len(projectors), self.dimension)
        
        # Verify projector properties
        for projector in projectors:
            # Check shape
            self.assertEqual(projector.shape, (self.dimension, self.dimension))
            
            # Check Hermiticity
            self.assertTrue(np.allclose(projector, projector.conj().T))
            
            # Check idempotency (P² = P)
            self.assertTrue(np.allclose(
                np.matmul(projector, projector),
                projector
            ))
    
    def test_collapse_operation(self):
        """Test collapse operation on wave function."""
        # Create superposition state
        state = np.ones(self.dimension) / np.sqrt(self.dimension)
        self.wave_function.set_state(state)
        
        # Apply collapse
        outcome, probability = self.collapse_operator.apply_collapse(
            self.wave_function
        )
        
        # Verify results
        self.assertIsInstance(outcome, int)
        self.assertGreaterEqual(outcome, 0)
        self.assertLess(outcome, self.dimension)
        
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        
        # Verify collapsed state
        final_state = self.wave_function.amplitudes
        self.assertTrue(np.allclose(
            np.abs(final_state[outcome])**2,
            1.0
        ))

class TestContingencyCollapse(unittest.TestCase):
    """Test cases for contingency collapse analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 2
        self.analyzer = ContingencyCollapse(self.num_qubits)
        self.wave_function = WaveFunction(self.num_qubits)
    
    def test_collapse_patterns(self):
        """Test analysis of collapse patterns."""
        # Create test state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.wave_function.set_state(state)
        
        # Analyze patterns
        metrics = self.analyzer.analyze_collapse_patterns(
            self.wave_function,
            num_trials=100
        )
        
        # Verify metrics
        self.assertIn('collapse_entropy', metrics)
        self.assertIn('average_probability', metrics)
        self.assertIn('probability_variance', metrics)
        self.assertIn('pattern_strength', metrics)
        self.assertIn('contingency_measure', metrics)
        
        # Check metric ranges
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
    
    def test_collapse_entropy(self):
        """Test collapse entropy calculation."""
        # Create test outcomes
        outcomes = [0, 1, 0, 1, 2, 3, 2, 3]
        
        # Calculate entropy
        entropy = self.analyzer._calculate_collapse_entropy(outcomes)
        
        # Verify entropy
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
    
    def test_pattern_strength(self):
        """Test pattern strength analysis."""
        # Create test outcomes with pattern
        outcomes = [0, 1, 0, 1, 0, 1, 0, 1]
        
        # Analyze pattern strength
        strength = self.analyzer._analyze_pattern_strength(outcomes)
        
        # Verify strength
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
        
        # Pattern should be detected
        self.assertGreater(strength, 0.5)
    
    def test_contingency_measure(self):
        """Test contingency measure calculation."""
        # Create test data
        outcomes = [0, 1, 0, 1]
        probabilities = [0.5, 0.5, 0.5, 0.5]
        
        # Calculate contingency
        contingency = self.analyzer._measure_contingency(
            outcomes,
            probabilities
        )
        
        # Verify measure
        self.assertGreaterEqual(contingency, 0.0)
        self.assertLessEqual(contingency, 1.0)

class TestNecessityAnalysis(unittest.TestCase):
    """Test cases for necessity analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 2
        self.wave_function = WaveFunction(self.num_qubits)
        self.analyzer = ContingencyCollapse(self.num_qubits)
    
    def test_collapse_necessity(self):
        """Test analysis of collapse necessity."""
        # Create test state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.wave_function.set_state(state)
        
        # Analyze necessity
        analysis = analyze_collapse_necessity(
            self.wave_function,
            num_trials=100
        )
        
        # Verify analysis structure
        self.assertIn('collapse_determinism', analysis)
        self.assertIn('pattern_necessity', analysis)
        self.assertIn('contingency_level', analysis)
        self.assertIn('predictability', analysis)
        self.assertIn('necessity_score', analysis)
        
        # Check score ranges
        for metric in analysis.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid dimension
        with self.assertRaises(ValueError):
            CollapseOperator(0)
        
        # Test invalid number of trials
        with self.assertRaises(ValueError):
            self.analyzer.analyze_collapse_patterns(
                self.wave_function,
                num_trials=0
            )
        
        # Test invalid wave function
        with self.assertRaises(ValueError):
            analyze_collapse_necessity(None)

if __name__ == '__main__':
    unittest.main()