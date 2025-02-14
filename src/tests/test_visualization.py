"""
Tests for visualization utilities.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..utils.visualization import (
    QuantumVisualizer,
    ModalLogicVisualizer,
    plot_results
)

class TestQuantumVisualizer(unittest.TestCase):
    """Test cases for quantum visualization."""
    
    def setUp(self):
        """Set up test environment."""
        self.visualizer = QuantumVisualizer()
        plt.close('all')  # Close any existing plots
    
    def test_measurement_probabilities(self):
        """Test probability distribution plotting."""
        # Create test probabilities
        probabilities = {
            '00': 0.5,
            '01': 0.3,
            '10': 0.1,
            '11': 0.1
        }
        
        # Create plot
        fig = self.visualizer.plot_measurement_probabilities(probabilities)
        
        # Verify plot
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Verify plot elements
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), len(probabilities))
        
        plt.close(fig)
    
    def test_necessity_analysis(self):
        """Test necessity analysis plotting."""
        # Create test metrics
        metrics = {
            'entropy': 0.7,
            'stability': 0.8,
            'coherence': 0.6,
            'causality': 0.9
        }
        
        # Create plot
        fig = self.visualizer.plot_necessity_analysis(metrics)
        
        # Verify plot
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Verify plot elements
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), len(metrics))
        
        plt.close(fig)
    
    def test_quantum_evolution(self):
        """Test quantum evolution plotting."""
        # Create test data
        num_steps = 10
        states = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, np.pi/2, num_steps)]
        times = list(range(num_steps))
        
        # Create plot
        fig = self.visualizer.plot_quantum_evolution(states, times)
        
        # Verify plot
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Verify plot elements
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)  # One line per basis state
        
        plt.close(fig)
    
    def test_bloch_sphere(self):
        """Test Bloch sphere plotting."""
        # Create test state
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Create plot
        fig = self.visualizer.plot_bloch_sphere(state)
        
        # Verify plot
        self.assertIsInstance(fig, Figure)
        
        plt.close(fig)

class TestModalLogicVisualizer(unittest.TestCase):
    """Test cases for modal logic visualization."""
    
    def setUp(self):
        """Set up test environment."""
        self.visualizer = ModalLogicVisualizer()
        plt.close('all')
    
    def test_modal_graph(self):
        """Test modal logic graph plotting."""
        # Create test data
        worlds = {
            'w1': {'w2', 'w3'},
            'w2': {'w3'},
            'w3': set()
        }
        properties = {
            'w1': {'P', 'Q'},
            'w2': {'P'},
            'w3': {'Q'}
        }
        
        # Create plot
        fig = self.visualizer.plot_modal_graph(worlds, properties)
        
        # Verify plot
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Verify plot elements
        ax = fig.axes[0]
        self.assertEqual(len(ax.texts), len(worlds))  # One text per world
        
        plt.close(fig)
    
    def test_graph_layout(self):
        """Test graph layout creation."""
        # Create test worlds
        worlds = {
            'w1': {'w2', 'w3'},
            'w2': {'w3'},
            'w3': set()
        }
        
        # Create layout
        positions = self.visualizer._create_graph_layout(worlds)
        
        # Verify layout
        self.assertEqual(len(positions), len(worlds))
        for pos in positions.values():
            self.assertEqual(len(pos), 2)  # 2D coordinates
            self.assertTrue(isinstance(pos[0], float))
            self.assertTrue(isinstance(pos[1], float))

class TestResultPlotting(unittest.TestCase):
    """Test cases for comprehensive result plotting."""
    
    def setUp(self):
        """Set up test environment."""
        self.visualizer = QuantumVisualizer()
        plt.close('all')
    
    def test_plot_results(self):
        """Test comprehensive result plotting."""
        # Create test results
        results = {
            'probabilities': {
                '00': 0.5,
                '11': 0.5
            },
            'necessity_metrics': {
                'entropy': 0.7,
                'stability': 0.8
            },
            'evolution': {
                'states': [np.array([1, 0]), np.array([0, 1])],
                'times': [0, 1]
            }
        }
        
        # Create temporary directory for plots
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Plot results
            plot_results(results, tmpdir)
            
            # Verify plot files were created
            expected_files = ['measurements.png', 'necessity.png', 'evolution.png']
            for filename in expected_files:
                filepath = os.path.join(tmpdir, filename)
                self.assertTrue(os.path.exists(filepath))
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid results dictionary
        with self.assertRaises(ValueError):
            plot_results({})
        
        # Test invalid probabilities
        with self.assertRaises(ValueError):
            self.visualizer.plot_measurement_probabilities({})
        
        # Test invalid metrics
        with self.assertRaises(ValueError):
            self.visualizer.plot_necessity_analysis({})
        
        # Test invalid evolution data
        with self.assertRaises(ValueError):
            self.visualizer.plot_quantum_evolution([], [])
        
        # Test invalid state vector
        with self.assertRaises(ValueError):
            self.visualizer.plot_bloch_sphere(np.array([1, 0, 0]))

if __name__ == '__main__':
    unittest.main()