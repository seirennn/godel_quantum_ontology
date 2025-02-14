"""
Tests for quantum entanglement simulations.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from ..quantum.entanglement import (
    create_entangled_state,
    measure_entanglement_strength,
    simulate_causal_dependencies,
    analyze_causal_structure
)

class TestQuantumEntanglement(unittest.TestCase):
    """Test cases for quantum entanglement operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 3
        self.depth = 2
    
    def test_create_entangled_state(self):
        """Test creation of entangled quantum states."""
        # Test chain entanglement
        circuit = create_entangled_state(self.num_qubits, 'chain')
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        
        # Test star entanglement
        circuit = create_entangled_state(self.num_qubits, 'star')
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertTrue(any(inst[0].name == 'cx' for inst in circuit.data))
        
        # Test fully connected entanglement
        circuit = create_entangled_state(self.num_qubits, 'fully_connected')
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertTrue(any(inst[0].name == 'cx' for inst in circuit.data))
    
    def test_measure_entanglement_strength(self):
        """Test measurement of entanglement strength."""
        # Create entangled state
        circuit = create_entangled_state(self.num_qubits, 'chain')
        
        # Test with default pairs
        metrics = measure_entanglement_strength(circuit)
        self.assertIn('average_entanglement', metrics)
        self.assertGreaterEqual(metrics['average_entanglement'], 0)
        self.assertLessEqual(metrics['average_entanglement'], 1)
        
        # Test with specific pairs
        pairs = [(0, 1), (1, 2)]
        metrics = measure_entanglement_strength(circuit, pairs)
        self.assertIn(f'pair_0_1', metrics)
        self.assertIn(f'pair_1_2', metrics)
    
    def test_simulate_causal_dependencies(self):
        """Test simulation of causal dependencies."""
        # Create base circuit
        circuit = create_entangled_state(self.num_qubits, 'chain')
        
        # Simulate dependencies
        modified_circuit = simulate_causal_dependencies(circuit, self.depth)
        self.assertGreater(len(modified_circuit.data), len(circuit.data))
        
        # Verify circuit structure
        gate_types = set(inst[0].name for inst in modified_circuit.data)
        self.assertIn('cx', gate_types)  # Entangling gates
        self.assertIn('rz', gate_types)  # Phase rotations
    
    def test_analyze_causal_structure(self):
        """Test analysis of causal structure."""
        # Create and simulate circuit
        circuit = create_entangled_state(self.num_qubits, 'chain')
        circuit = simulate_causal_dependencies(circuit, self.depth)
        
        # Analyze structure
        metrics = analyze_causal_structure(circuit)
        
        # Verify metrics
        self.assertIn('global_entanglement', metrics)
        self.assertIn('average_local_causality', metrics)
        self.assertIn('causality_variance', metrics)
        self.assertIn('estimated_causal_depth', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['global_entanglement'], 0)
        self.assertGreaterEqual(metrics['average_local_causality'], 0)
        self.assertGreaterEqual(metrics['causality_variance'], 0)
        self.assertEqual(metrics['estimated_causal_depth'], circuit.depth())
    
    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        patterns = ['chain', 'star', 'fully_connected']
        
        for pattern in patterns:
            # Create circuit with pattern
            circuit = create_entangled_state(self.num_qubits, pattern)
            
            # Measure entanglement
            metrics = measure_entanglement_strength(circuit)
            
            # Verify pattern-specific properties
            if pattern == 'chain':
                # Chain should have strong nearest-neighbor entanglement
                self.assertGreater(metrics['pair_0_1'], 0.1)
            elif pattern == 'star':
                # Star should have strong central entanglement
                central_pairs = [f'pair_0_{i}' for i in range(1, self.num_qubits)]
                for pair in central_pairs:
                    self.assertIn(pair, metrics)
            elif pattern == 'fully_connected':
                # Fully connected should have many entangled pairs
                self.assertGreater(
                    len([k for k in metrics.keys() if k.startswith('pair_')]),
                    self.num_qubits - 1
                )
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid number of qubits
        with self.assertRaises(ValueError):
            create_entangled_state(0, 'chain')
        
        # Test invalid pattern
        with self.assertRaises(ValueError):
            create_entangled_state(self.num_qubits, 'invalid_pattern')
        
        # Test invalid qubit pairs
        circuit = create_entangled_state(self.num_qubits, 'chain')
        with self.assertRaises(ValueError):
            measure_entanglement_strength(circuit, [(0, self.num_qubits + 1)])
        
        # Test invalid depth
        with self.assertRaises(ValueError):
            simulate_causal_dependencies(circuit, -1)
    
    def test_entanglement_persistence(self):
        """Test persistence of entanglement through operations."""
        # Create initial entangled state
        circuit = create_entangled_state(self.num_qubits, 'chain')
        initial_metrics = measure_entanglement_strength(circuit)
        
        # Apply operations
        modified_circuit = simulate_causal_dependencies(circuit, self.depth)
        final_metrics = measure_entanglement_strength(modified_circuit)
        
        # Verify entanglement persists
        self.assertGreater(final_metrics['average_entanglement'], 0)
        
        # Compare with initial entanglement
        self.assertNotEqual(
            initial_metrics['average_entanglement'],
            final_metrics['average_entanglement']
        )

if __name__ == '__main__':
    unittest.main()