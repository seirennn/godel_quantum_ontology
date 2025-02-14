"""
Physics module for the Divine Algorithm project.
"""

from .wave_function import (
    WaveFunction,
    ContingencyAnalyzer,
    analyze_wave_function
)
from .collapse import (
    CollapseOperator,
    ContingencyCollapse,
    analyze_collapse_necessity
)

__all__ = [
    'WaveFunction',
    'ContingencyAnalyzer',
    'analyze_wave_function',
    'CollapseOperator',
    'ContingencyCollapse',
    'analyze_collapse_necessity'
]