"""
Test suite for the Divine Algorithm project.
"""

from .test_lambda_calc import TestLambdaCalculus
from .test_modal import TestModalLogic
from .test_circuits import TestQuantumCircuits
from .test_entanglement import TestQuantumEntanglement
from .test_measurement import TestQuantumMeasurement
from .test_wave_function import TestWaveFunction, TestContingencyAnalyzer
from .test_collapse import TestCollapseOperator, TestContingencyCollapse, TestNecessityAnalysis
from .test_visualization import TestQuantumVisualizer, TestModalLogicVisualizer, TestResultPlotting
from .test_analysis import TestResultAnalyzer, TestNecessityAnalyzer

__all__ = [
    'TestQuantumCircuits',
    'TestQuantumEntanglement',
    'TestQuantumMeasurement',
    'TestModalLogic',
    'TestLambdaCalculus',
    'TestWaveFunction',
    'TestContingencyAnalyzer',
    'TestCollapseOperator',
    'TestContingencyCollapse',
    'TestNecessityAnalysis',
    'TestQuantumVisualizer',
    'TestModalLogicVisualizer',
    'TestResultPlotting',
    'TestResultAnalyzer',
    'TestNecessityAnalyzer'
]