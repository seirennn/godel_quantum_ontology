"""
Tests for data analysis utilities.
"""

import unittest
import numpy as np
import pandas as pd
from ..utils.analysis import (
    ResultAnalyzer,
    NecessityAnalyzer,
    analyze_results
)

class TestResultAnalyzer(unittest.TestCase):
    """Test cases for result analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = ResultAnalyzer()
    
    def test_analyze_quantum_results(self):
        """Test analysis of quantum results."""
        # Create test results
        results = {
            'measurements': {
                '00': 0.5,
                '11': 0.5
            },
            'necessity_metrics': {
                'entropy': 0.7,
                'stability': 0.8,
                'coherence': 0.6
            },
            'evolution': {
                'states': [np.array([1, 0]), np.array([0, 1])],
                'times': [0, 1]
            }
        }
        
        # Analyze results
        analysis = self.analyzer.analyze_quantum_results(results)
        
        # Verify analysis structure
        self.assertIn('measurement_analysis', analysis)
        self.assertIn('necessity_analysis', analysis)
        self.assertIn('evolution_analysis', analysis)
    
    def test_analyze_measurements(self):
        """Test measurement analysis."""
        # Create test measurements
        measurements = {
            '00': 0.4,
            '01': 0.3,
            '10': 0.2,
            '11': 0.1
        }
        
        # Analyze measurements
        analysis = self.analyzer.analyze_measurements(measurements)
        
        # Verify analysis metrics
        self.assertIn('mean', analysis)
        self.assertIn('std', analysis)
        self.assertIn('entropy', analysis)
        self.assertIn('normality_p_value', analysis)
        
        # Check metric ranges
        for metric in analysis.values():
            self.assertIsInstance(metric, (float, np.float64))
    
    def test_analyze_necessity(self):
        """Test necessity analysis."""
        # Create test metrics
        results = {
            'measurements': {
                'entropy': 0.7,
                'stability': 0.8,
                'coherence': 0.6,
                'causality': 0.9
            }
        }
        
        # Analyze necessity
        analysis = self.analyzer.analyze_necessity(results)
        
        # Verify analysis structure
        self.assertIn('measurement_stability', analysis)
        self.assertIn('confidence', analysis)
        
        # Check metric ranges
        for metric in analysis.values():
            self.assertIsInstance(metric, (float, np.float64))
    
    def test_analyze_evolution(self):
        """Test evolution analysis."""
        # Create test evolution data
        states = [
            np.array([1, 0]),
            np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
            np.array([0, 1])
        ]
        
        # Analyze evolution
        analysis = self.analyzer.analyze_evolution(states)
        
        # Verify analysis metrics
        self.assertIn('mean_change', analysis)
        self.assertIn('max_change', analysis)
        self.assertIn('stability', analysis)
        
        # Check metric ranges
        for metric in analysis.values():
            self.assertIsInstance(metric, (float, np.float64))

class TestNecessityAnalyzer(unittest.TestCase):
    """Test cases for necessity analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = NecessityAnalyzer()
    
    def test_analyze_necessity_proof(self):
        """Test analysis of necessity proof."""
        # Create test results
        quantum_results = {
            'measurements': {'00': 0.5, '11': 0.5},
            'evolution': {'states': [np.array([1, 0])], 'times': [0]}
        }
        modal_results = {
            'necessity_patterns': {'pattern1': 0.8},
            'contingency_patterns': {'pattern1': 0.2}
        }
        
        # Analyze proof
        analysis = self.analyzer.analyze_necessity_proof(
            quantum_results,
            modal_results
        )
        
        # Verify analysis structure
        self.assertIn('quantum_evidence', analysis)
        self.assertIn('modal_implications', analysis)
        self.assertIn('proof_strength', analysis)
    
    def test_analyze_quantum_evidence(self):
        """Test quantum evidence analysis."""
        # Create test results
        results = {
            'measurements': {'00': 0.5, '11': 0.5},
            'evolution': {'states': [np.array([1, 0])], 'times': [0]}
        }
        
        # Analyze evidence
        evidence = self.analyzer._analyze_quantum_evidence(results)
        
        # Verify evidence metrics
        self.assertIn('measurement_support', evidence)
        self.assertIn('evolution_support', evidence)
        self.assertIn('evidence_strength', evidence)
    
    def test_analyze_modal_implications(self):
        """Test modal implications analysis."""
        # Create test results
        results = {
            'necessity_patterns': {'pattern1': 0.8},
            'contingency_patterns': {'pattern1': 0.2}
        }
        
        # Analyze implications
        implications = self.analyzer._analyze_modal_implications(results)
        
        # Verify implication metrics
        self.assertIn('pattern_strength', implications)
        self.assertIn('necessity_strength', implications)

def test_comprehensive_analysis():
    """Test comprehensive result analysis."""
    # Create test results
    results = {
        'quantum_results': {
            'measurements': {'00': 0.5, '11': 0.5},
            'necessity_metrics': {
                'entropy': 0.7,
                'stability': 0.8
            }
        },
        'modal_results': {
            'necessity_patterns': {'pattern1': 0.8},
            'contingency_patterns': {'pattern1': 0.2}
        }
    }
    
    # Perform comprehensive analysis
    analysis = analyze_results(results)
    
    # Verify analysis structure
    assert 'quantum_analysis' in analysis
    assert 'proof_analysis' in analysis

if __name__ == '__main__':
    unittest.main()