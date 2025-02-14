
"""
Visualization utilities for the Divine Algorithm project.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector

class QuantumVisualizer:
    """Visualization tools for quantum states and measurements."""
    
    def __init__(self, style: str = "darkgrid"):
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = [10, 6]
    
    def plot_measurement_probabilities(
        self,
        probabilities: Dict[str, float],
        title: str = "Quantum State Probabilities"
    ) -> Figure:
        """Plot measurement probability distribution."""
        if not probabilities:
            raise ValueError("Probabilities dictionary cannot be empty")
        if not all(isinstance(p, (int, float)) for p in probabilities.values()):
            raise ValueError("All probabilities must be numeric")
            
        fig, ax = plt.subplots()
        states = list(probabilities.keys())
        probs = list(probabilities.values())
        bars = ax.bar(states, probs)
        ax.set_xlabel("Quantum State")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.3f}", ha="center", va="bottom")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_necessity_analysis(
        self,
        metrics: Dict[str, float],
        title: str = "Necessity Analysis"
    ) -> Figure:
        """Plot necessity analysis metrics."""
        if not metrics:
            raise ValueError("Metrics dictionary cannot be empty")
        if not all(isinstance(v, (int, float)) for v in metrics.values()):
            raise ValueError("All metric values must be numeric")
            
        fig, ax = plt.subplots()
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_metrics)
        bars = ax.barh(labels, values)
        ax.set_xlabel("Metric Value")
        ax.set_title(title)
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f"{width:.3f}", ha="left", va="center")
        plt.tight_layout()
        return fig
    
    def plot_quantum_evolution(
        self,
        states: List[np.ndarray],
        times: List[float],
        title: str = "Quantum State Evolution"
    ) -> Figure:
        """Plot quantum state evolution over time."""
        if not states or not times:
            raise ValueError("States and times lists cannot be empty")
        if len(states) != len(times):
            raise ValueError("States and times must have same length")
            
        fig, ax = plt.subplots()
        probabilities = []
        for state in states:
            probs = np.abs(state) ** 2
            probabilities.append(probs)
        probabilities = np.array(probabilities)
        for i in range(probabilities.shape[1]):
            ax.plot(times, probabilities[:, i], label=f"State |{i}⟩")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_bloch_sphere(
        self,
        statevector: np.ndarray,
        title: str = "Quantum State Bloch Sphere"
    ) -> Figure:
        """Plot quantum state on Bloch sphere."""
        if len(statevector) != 2:
            raise ValueError("State vector must be 2-dimensional for Bloch sphere")
            
        qiskit_state = Statevector(statevector)
        fig = plot_bloch_multivector(qiskit_state)
        plt.title(title)
        return fig

class ModalLogicVisualizer:
    """Visualization tools for modal logic analysis."""
    
    def plot_modal_graph(
        self,
        worlds: Dict[str, Set[str]],
        properties: Dict[str, Set[str]],
        title: str = "Modal Logic Graph"
    ) -> Figure:
        """Plot modal logic graph."""
        if not worlds:
            raise ValueError("Worlds dictionary cannot be empty")
        if not properties:
            raise ValueError("Properties dictionary cannot be empty")
        if not all(w in properties for w in worlds):
            raise ValueError("Properties must be defined for all worlds")
            
        fig, ax = plt.subplots()
        pos = self._create_graph_layout(worlds)
        for world_id, position in pos.items():
            ax.plot(position[0], position[1], "o", markersize=20)
            ax.text(position[0], position[1],
                   f"{world_id}\n{properties[world_id]}",
                   ha="center", va="center")
        for world_id, accessible in worlds.items():
            start = pos[world_id]
            for target in accessible:
                end = pos[target]
                ax.arrow(start[0], start[1],
                        end[0] - start[0], end[1] - start[1],
                        head_width=0.1, length_includes_head=True)
        ax.set_title(title)
        ax.axis("equal")
        plt.tight_layout()
        return fig
    
    def _create_graph_layout(
        self,
        worlds: Dict[str, Set[str]]
    ) -> Dict[str, Tuple[float, float]]:
        """Create graph layout."""
        num_worlds = len(worlds)
        positions = {}
        for i, world_id in enumerate(worlds.keys()):
            angle = 2 * np.pi * i / num_worlds
            positions[world_id] = (np.cos(angle), np.sin(angle))
        return positions

def plot_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    """
    Plot comprehensive analysis results.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save plot files
    
    Raises:
        ValueError: If results is invalid or output_dir is not writable
    """
    # Validate inputs
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")
    if not results:
        raise ValueError("Results dictionary cannot be empty")
    if not isinstance(output_dir, str):
        raise ValueError("Output directory must be a string")
        
    # Validate required data
    if 'probabilities' in results and not isinstance(results['probabilities'], dict):
        raise ValueError("Probabilities must be a dictionary")
    if 'necessity_metrics' in results and not isinstance(results['necessity_metrics'], dict):
        raise ValueError("Necessity metrics must be a dictionary")
    if 'evolution' in results:
        if not isinstance(results['evolution'], dict):
            raise ValueError("Evolution data must be a dictionary")
        if 'states' not in results['evolution'] or 'times' not in results['evolution']:
            raise ValueError("Evolution data must contain 'states' and 'times'")
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = QuantumVisualizer()
    
    # Plot measurement probabilities
    if 'probabilities' in results:
        fig = visualizer.plot_measurement_probabilities(results['probabilities'])
        plt.savefig(os.path.join(output_dir, 'measurements.png'))
        plt.close(fig)
    
    # Plot necessity metrics
    if 'necessity_metrics' in results:
        fig = visualizer.plot_necessity_analysis(results['necessity_metrics'])
        plt.savefig(os.path.join(output_dir, 'necessity.png'))
        plt.close(fig)
    
    # Plot evolution data
    if 'evolution' in results:
        fig = visualizer.plot_quantum_evolution(
            results['evolution']['states'],
            results['evolution']['times']
        )
        plt.savefig(os.path.join(output_dir, 'evolution.png'))
        plt.close(fig)
