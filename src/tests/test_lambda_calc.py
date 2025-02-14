"""
Tests for lambda calculus operations.
"""

import unittest
from typing import Dict
from ..logic.lambda_calc import (
    Term,
    Variable,
    Abstraction,
    Application,
    LambdaCalculus,
    QuantumLogicAnalyzer
)

class TestLambdaCalculus(unittest.TestCase):
    """Test cases for lambda calculus operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.calc = LambdaCalculus()
        self.analyzer = QuantumLogicAnalyzer()
    
    def test_variable_term(self):
        """Test variable term operations."""
        # Create variable
        var = Variable('x')
        
        # Test evaluation
        self.assertEqual(var.evaluate(), var)
        
        # Test substitution
        y = Variable('y')
        substituted = var.substitute('x', y)
        self.assertEqual(substituted, y)
        
        # Test string representation
        self.assertEqual(str(var), 'x')
    
    def test_abstraction_term(self):
        """Test lambda abstraction operations."""
        # Create abstraction
        var = Variable('x')
        body = Variable('y')
        abs_term = Abstraction('x', body)
        
        # Test evaluation
        self.assertEqual(abs_term.evaluate(), abs_term)
        
        # Test substitution
        z = Variable('z')
        substituted = abs_term.substitute('y', z)
        self.assertEqual(str(substituted), 'λx.z')
        
        # Test variable capture avoidance
        substituted = abs_term.substitute('x', z)
        self.assertEqual(str(substituted), 'λx.y')
    
    def test_application_term(self):
        """Test function application operations."""
        # Create application
        func = Abstraction('x', Variable('x'))
        arg = Variable('y')
        app = Application(func, arg)
        
        # Test evaluation
        evaluated = app.evaluate()
        self.assertEqual(str(evaluated), 'y')
        
        # Test substitution
        z = Variable('z')
        substituted = app.substitute('y', z)
        self.assertEqual(
            str(substituted.evaluate()),
            'z'
        )
    
    def test_church_numerals(self):
        """Test Church numeral creation and operations."""
        # Create Church numerals
        zero = self.calc.create_church_numeral(0)
        one = self.calc.create_church_numeral(1)
        two = self.calc.create_church_numeral(2)
        
        # Test evaluation
        self.assertIsInstance(zero, Abstraction)
        self.assertIsInstance(one, Abstraction)
        self.assertIsInstance(two, Abstraction)
        
        # Test Church numeral structure
        self.assertEqual(
            str(zero),
            'λf.λx.x'
        )
        self.assertEqual(
            str(one),
            'λf.λx.f x'
        )
    
    def test_church_booleans(self):
        """Test Church boolean operations."""
        # Create Church booleans
        true_val = self.calc.create_boolean(True)
        false_val = self.calc.create_boolean(False)
        
        # Test evaluation
        self.assertIsInstance(true_val, Abstraction)
        self.assertIsInstance(false_val, Abstraction)
        
        # Test boolean structure
        self.assertEqual(
            str(true_val),
            'λt.λf.t'
        )
        self.assertEqual(
            str(false_val),
            'λt.λf.f'
        )
    
    def test_quantum_logic_analysis(self):
        """Test quantum logic analysis."""
        # Create test term
        var = Variable('x')
        term = Application(var, var)
        
        # Analyze term
        analysis = self.analyzer.analyze_quantum_logic(term)
        
        # Verify analysis structure
        self.assertIn('free_vars', analysis)
        self.assertIn('complexity', analysis)
        self.assertIn('evaluated', analysis)
        self.assertIn('reducible', analysis)
    
    def test_quantum_logic_evaluation(self):
        """Test evaluation of quantum logic terms."""
        # Create test term
        var = Variable('x')
        term = Application(var, var)
        
        # Analyze and evaluate
        results = self.analyzer.analyze_quantum_logic(term)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('free_vars', results)
        self.assertIn('complexity', results)
        self.assertIn('evaluated', results)
        self.assertIn('reducible', results)
    
    def test_complex_terms(self):
        """Test complex lambda calculus terms."""
        # Create complex nested term
        x = Variable('x')
        y = Variable('y')
        inner_abs = Abstraction('y', Application(y, x))
        outer_abs = Abstraction('x', inner_abs)
        
        # Test evaluation
        evaluated = outer_abs.evaluate()
        self.assertEqual(str(evaluated), 'λx.λy.y x')
        
        # Test substitution
        z = Variable('z')
        substituted = outer_abs.substitute('x', z)
        self.assertEqual(str(substituted), 'λx.λy.y x')
    
    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        # Test invalid Church numeral
        with self.assertRaises(ValueError):
            self.calc.create_church_numeral(-1)
        
        # Test invalid term substitution
        var = Variable('x')
        with self.assertRaises(TypeError):
            var.substitute('x', 'not_a_term')
        
        # Test invalid term analysis
        with self.assertRaises(AttributeError):
            self.analyzer.analyze_quantum_logic(None)

if __name__ == '__main__':
    unittest.main()