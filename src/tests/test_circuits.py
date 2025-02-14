"""
Tests for quantum circuit implementations.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from ..quantum.circuits import (
    create_contingency_circuit,
    add_causal_layer,
    create_measurement_circuit,
    create_necessity_test_circuit,
    analyze_circuit_complexity
)

class TestQuantumCircuits(unittest.TestCase):
    """Test cases for quantum circuit operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_qubits = 3
        self.depth = 2
        self.backend = Aer.get_backend('statevector_simulator')
        self.sampler = Sampler()
    
    def test_create_contingency_circuit(self):
        """Test creation of contingency circuit."""
        # Test with default initialization
        circuit = create_contingency_circuit(self.num_qubits, self.depth)
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        self.assertIsInstance(circuit, QuantumCircuit)
        
        # Test with custom initial state
        initial_state = np.ones(2**self.num_qubits) / np.sqrt(2**self.num_qubits)
        circuit = create_contingency_circuit(
            self.num_qubits,
            self.depth,
            initial_state
        )
        self.assertEqual(circuit.num_qubits, self.num_qubits)
    
    def test_add_causal_layer(self):
        """Test adding causal layer to circuit."""
        # Test standard causal layer
        base_circuit = create_contingency_circuit(self.num_qubits, self.depth)
        circuit = add_causal_layer(base_circuit.copy(), 'standard')
        self.assertGreater(len(circuit.data), len(base_circuit.data))
        
        # Test complex causal layer
        base_circuit = create_contingency_circuit(self.num_qubits, self.depth)
        circuit = add_causal_layer(base_circuit.copy(), 'complex')
        self.assertGreater(len(circuit.data), len(base_circuit.data))
        
        # Test quantum walk layer
        base_circuit = create_contingency_circuit(self.num_qubits, self.depth)
        circuit = add_causal_layer(base_circuit.copy(), 'quantum_walk')
        self.assertGreater(len(circuit.data), len(base_circuit.data))
    
    def test_create_measurement_circuit(self):
        """Test creation of measurement circuit."""
        base_circuit = create_contingency_circuit(self.num_qubits, self.depth)
        
        # Test computational basis measurement
        circuit = create_measurement_circuit(base_circuit, 'computational')
        self.assertTrue(any(inst[0].name == 'measure' for inst in circuit.data))
        
        # Test Hadamard basis measurement
        circuit = create_measurement_circuit(base_circuit, 'hadamard')
        self.assertTrue(any(inst[0].name == 'h' for inst in circuit.data))
        self.assertTrue(any(inst[0].name == 'measure' for inst in circuit.data))
    
    def test_create_necessity_test_circuit(self):
        """Test creation of necessity test circuit."""
        circuit = create_necessity_test_circuit(self.num_qubits)
        
        # Verify circuit properties
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        self.assertTrue(any(inst[0].name == 'measure' for inst in circuit.data))
        
        # Execute circuit and verify output
        job = self.sampler.run([circuit])
        result = job.result()
        self.assertIsNotNone(result)
    
    def test_analyze_circuit_complexity(self):
        """Test circuit complexity analysis."""
        circuit = create_necessity_test_circuit(self.num_qubits)
        complexity = analyze_circuit_complexity(circuit)
        
        # Verify complexity metrics
        self.assertIn('total_gates', complexity)
        self.assertIn('gate_distribution', complexity)
        self.assertIn('entanglement_operations', complexity)
        self.assertIn('circuit_width', complexity)
        self.assertIn('circuit_depth', complexity)
        
        # Verify metric values
        self.assertGreater(complexity['total_gates'], 0)
        self.assertEqual(complexity['circuit_width'], self.num_qubits)
    
    def test_circuit_execution(self):
        """Test execution of complete circuit pipeline."""
        # Create and modify circuit
        circuit = create_contingency_circuit(self.num_qubits, self.depth)
        circuit = add_causal_layer(circuit, 'standard')
        circuit = create_measurement_circuit(circuit, 'computational')
        
        # Execute circuit
        job = self.sampler.run([circuit])
        result = job.result()
        
        # Verify execution results
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'quasi_dists'))
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid number of qubits
        with self.assertRaises(ValueError):
            create_contingency_circuit(0, self.depth)
        
        # Test invalid depth
        with self.assertRaises(ValueError):
            create_contingency_circuit(self.num_qubits, -1)
        
        # Test invalid layer type
        base_circuit = create_contingency_circuit(self.num_qubits, self.depth)
        with self.assertRaises(ValueError):
            add_causal_layer(base_circuit, 'invalid_type')
        
        # Test invalid measurement basis
        with self.assertRaises(ValueError):
            create_measurement_circuit(base_circuit, 'invalid_basis')

if __name__ == '__main__':
    unittest.main()