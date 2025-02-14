"""
Logic module for the Divine Algorithm project.
"""

from .modal import (
    ModalOperator,
    World,
    ModalFrame,
    ModalLogic,
    analyze_quantum_necessity
)
from .lambda_calc import (
    Term,
    Variable,
    Abstraction,
    Application,
    LambdaCalculus,
    QuantumLogicAnalyzer
)

__all__ = [
    'ModalOperator',
    'World',
    'ModalFrame',
    'ModalLogic',
    'analyze_quantum_necessity',
    'Term',
    'Variable',
    'Abstraction',
    'Application',
    'LambdaCalculus',
    'QuantumLogicAnalyzer'
]