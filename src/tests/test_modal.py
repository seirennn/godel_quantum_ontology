"""
Tests for modal logic operations.
"""

import unittest
from typing import Set
from ..logic.modal import (
    ModalOperator,
    World,
    ModalFrame,
    ModalLogic,
    analyze_quantum_necessity
)

class TestModalLogic(unittest.TestCase):
    """Test cases for modal logic operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.frame = ModalFrame()
        
        # Add test worlds
        self.frame.add_world('w1', {'P', 'Q'})
        self.frame.add_world('w2', {'P'})
        self.frame.add_world('w3', {'Q'})
        
        # Add accessibility relations
        self.frame.add_accessibility('w1', 'w2')
        self.frame.add_accessibility('w1', 'w3')
        self.frame.add_accessibility('w2', 'w3')
        
        # Set actual world
        self.frame.set_actual_world('w1')
        
        # Create modal logic analyzer
        self.logic = ModalLogic(self.frame)
    
    def test_modal_operators(self):
        """Test modal operators."""
        # Test operator creation
        self.assertIsInstance(ModalOperator.NECESSARY, ModalOperator)
        self.assertIsInstance(ModalOperator.POSSIBLE, ModalOperator)
        self.assertIsInstance(ModalOperator.CONTINGENT, ModalOperator)
        self.assertIsInstance(ModalOperator.IMPOSSIBLE, ModalOperator)
        
        # Test operator distinctness
        operators = set(ModalOperator)
        self.assertEqual(len(operators), 4)
    
    def test_world_creation(self):
        """Test world creation and properties."""
        world = World('test', {'A', 'B'}, {'w1', 'w2'})
        
        self.assertEqual(world.id, 'test')
        self.assertEqual(world.properties, {'A', 'B'})
        self.assertEqual(world.accessible_worlds, {'w1', 'w2'})
    
    def test_modal_frame(self):
        """Test modal frame operations."""
        # Test world addition
        self.frame.add_world('w4', {'R'})
        self.assertIn('w4', self.frame.worlds)
        
        # Test accessibility addition
        self.frame.add_accessibility('w4', 'w1')
        self.assertIn('w1', self.frame.worlds['w4'].accessible_worlds)
        
        # Test actual world setting
        self.frame.set_actual_world('w4')
        self.assertEqual(self.frame.actual_world, 'w4')
    
    def test_necessity_evaluation(self):
        """Test evaluation of necessity."""
        # Test necessity of property P
        is_necessary = self.logic.evaluate_necessity('P', 'w1')
        self.assertIsInstance(is_necessary, bool)
        
        # Test necessity in different worlds
        is_necessary_w2 = self.logic.evaluate_necessity('P', 'w2')
        self.assertIsInstance(is_necessary_w2, bool)
    
    def test_possibility_evaluation(self):
        """Test evaluation of possibility."""
        # Test possibility of property Q
        is_possible = self.logic.evaluate_possibility('Q', 'w1')
        self.assertTrue(is_possible)  # Q is true in w1 and w3
        
        # Test impossible property
        is_possible = self.logic.evaluate_possibility('R', 'w1')
        self.assertFalse(is_possible)  # R is not true in any accessible world
    
    def test_contingency_evaluation(self):
        """Test evaluation of contingency."""
        # Test contingency of property Q
        is_contingent = self.logic.evaluate_contingency('Q', 'w1')
        self.assertTrue(is_contingent)  # Q is possible but not necessary
        
        # Test non-contingent property
        is_contingent = self.logic.evaluate_contingency('R', 'w1')
        self.assertFalse(is_contingent)  # R is impossible, thus not contingent
    
    def test_modal_operator_determination(self):
        """Test determination of modal operators."""
        # Test necessary property
        operator = self.logic.get_modal_operator('P', 'w2')
        self.assertEqual(operator, ModalOperator.NECESSARY)
        
        # Test contingent property
        operator = self.logic.get_modal_operator('Q', 'w1')
        self.assertEqual(operator, ModalOperator.CONTINGENT)
        
        # Test impossible property
        operator = self.logic.get_modal_operator('R', 'w1')
        self.assertEqual(operator, ModalOperator.IMPOSSIBLE)
    
    def test_quantum_necessity_analysis(self):
        """Test analysis of quantum necessity."""
        # Create mock quantum measurement results
        measurement_results = {
            'probabilities': {
                '00': 0.5,
                '11': 0.5
            }
        }
        
        # Analyze necessity
        analysis = analyze_quantum_necessity(measurement_results)
        
        # Verify analysis structure
        self.assertIn('necessary_states', analysis)
        self.assertIn('contingent_states', analysis)
        self.assertIn('impossible_states', analysis)
        self.assertIn('necessity_ratio', analysis)
        self.assertIn('contingency_ratio', analysis)
        
        # Check ratio ranges
        self.assertGreaterEqual(analysis['necessity_ratio'], 0.0)
        self.assertLessEqual(analysis['necessity_ratio'], 1.0)
        self.assertGreaterEqual(analysis['contingency_ratio'], 0.0)
        self.assertLessEqual(analysis['contingency_ratio'], 1.0)
    
    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        # Test invalid world access
        with self.assertRaises(ValueError):
            self.logic.evaluate_necessity('P', 'invalid_world')
        
        # Test invalid accessibility
        with self.assertRaises(ValueError):
            self.frame.add_accessibility('w1', 'invalid_world')
        
        # Test invalid actual world
        with self.assertRaises(ValueError):
            self.frame.set_actual_world('invalid_world')
    
    def test_complex_necessity_patterns(self):
        """Test complex patterns of necessity."""
        # Create more complex frame
        complex_frame = ModalFrame()
        
        # Add worlds with complex properties
        for i in range(5):
            properties = {f'P{j}' for j in range(i)}
            complex_frame.add_world(f'w{i}', properties)
            
        # Add complex accessibility relations
        for i in range(4):
            complex_frame.add_accessibility(f'w{i}', f'w{i+1}')
        
        # Test necessity patterns
        logic = ModalLogic(complex_frame)
        complex_frame.set_actual_world('w0')
        
        # Verify increasing property presence
        for i in range(5):
            prop = f'P{i}'
            is_possible = logic.evaluate_possibility(prop, 'w0')
            self.assertEqual(is_possible, i < 4)

if __name__ == '__main__':
    unittest.main()