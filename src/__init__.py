"""
Divine Algorithm: A Quantum Computational Proof of God's Existence
==============================================================

This package implements a quantum computational approach to exploring
the necessity of a self-existent being through simulation and analysis
of reality's fundamental structure.

Modules:
    quantum: Quantum computing implementations
    logic: Modal logic and lambda calculus operations
    physics: Wave function and collapse mechanisms
    utils: Visualization and analysis utilities
"""

from .quantum import *
from .logic import *
from .physics import *
from .utils import *

__version__ = '0.1.0'
__author__ = 'Gödel Quantum Research Team'
__license__ = 'MIT'

__all__ = (
    quantum.__all__ +
    logic.__all__ +
    physics.__all__ +
    utils.__all__
)