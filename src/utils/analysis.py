"""
Analysis utilities for the Divine Algorithm project.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats

class ResultAnalyzer:
    """Analyzer for quantum measurement results."""
    
    def analyze_measurements(self, results: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze quantum measurement results.
        
        Args:
            results: Dictionary of measurement results
            
        Returns:
            Dictionary of analysis metrics
        """
        metrics = {}
        values = list(results.values())
        
        # Calculate basic statistics
        metrics['mean'] = np.mean(values)
        metrics['std'] = np.std(values)
        metrics['entropy'] = self._calculate_entropy(values)
        
        # Perform normality test
        _, p_value = stats.normaltest(values)
        metrics['normality_p_value'] = p_value
        
        return metrics
    
    def analyze_evolution(self, states: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze quantum state evolution.
        
        Args:
            states: List of state vectors
            
        Returns:
            Dictionary of evolution metrics
        """
        metrics = {}
        
        # Calculate state changes
        changes = []
        for i in range(len(states)-1):
            change = np.linalg.norm(states[i+1] - states[i])
            changes.append(change)
        
        metrics['mean_change'] = np.mean(changes)
        metrics['max_change'] = np.max(changes)
        metrics['stability'] = 1.0 / (1.0 + np.std(changes))
        
        return metrics
    
    def analyze_necessity(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze necessity indicators in results.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Dictionary of necessity metrics
        """
        metrics = {}
        
        # Analyze measurement stability
        if 'measurements' in results:
            m_analysis = self.analyze_measurements(results['measurements'])
            metrics['measurement_stability'] = 1.0 / (1.0 + m_analysis['std'])
        
        # Analyze evolution stability
        if 'evolution' in results:
            e_analysis = self.analyze_evolution(results['evolution']['states'])
            metrics['evolution_stability'] = e_analysis['stability']
        
        # Calculate overall confidence
        metrics['confidence'] = self._calculate_confidence(metrics)
        
        return metrics
    
    def analyze_quantum_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of quantum results.
        
        Args:
            results: Dictionary of quantum results
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Analyze measurements
        if 'measurements' in results:
            analysis['measurement_analysis'] = self.analyze_measurements(
                results['measurements']
            )
        
        # Analyze evolution
        if 'evolution' in results:
            analysis['evolution_analysis'] = self.analyze_evolution(
                results['evolution']['states']
            )
        
        # Analyze necessity indicators
        analysis['necessity_analysis'] = self.analyze_necessity(results)
        
        return analysis
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy of probability distribution."""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate overall confidence score."""
        weights = {
            'measurement_stability': 0.4,
            'evolution_stability': 0.6
        }
        
        confidence = 0.0
        for key, weight in weights.items():
            if key in metrics:
                confidence += weight * metrics[key]
        
        return confidence

class NecessityAnalyzer:
    """Analyzer for necessity proofs."""
    
    def analyze_necessity_proof(
        self,
        quantum_results: Dict[str, Any],
        modal_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze necessity proof results.
        
        Args:
            quantum_results: Dictionary of quantum analysis results
            modal_results: Dictionary of modal logic results
            
        Returns:
            Dictionary of proof analysis results
        """
        analysis = {}
        
        # Analyze quantum evidence
        analysis['quantum_evidence'] = self._analyze_quantum_evidence(
            quantum_results
        )
        
        # Analyze modal implications
        analysis['modal_implications'] = self._analyze_modal_implications(
            modal_results
        )
        
        # Calculate overall strength
        analysis['proof_strength'] = self._calculate_proof_strength(analysis)
        
        return analysis
    
    def _analyze_quantum_evidence(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze quantum evidence for necessity."""
        evidence = {}
        
        # Calculate measurement support
        evidence['measurement_support'] = self._calculate_measurement_support(
            results.get('measurements', {})
        )
        
        # Calculate evolution support
        if 'evolution' in results:
            evidence['evolution_support'] = self._calculate_evolution_support(
                results['evolution']
            )
        
        # Calculate overall evidence strength
        evidence['evidence_strength'] = np.mean(list(evidence.values()))
        
        return evidence
    
    def _analyze_modal_implications(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze modal logic implications."""
        implications = {}
        
        # Analyze necessity patterns
        if 'necessity_patterns' in results:
            implications['pattern_strength'] = np.mean(
                list(results['necessity_patterns'].values())
            )
        
        # Analyze modal consistency
        if 'modal_consistency' in results:
            implications['consistency'] = results['modal_consistency']
        
        # Calculate overall implication strength
        implications['necessity_strength'] = np.mean(list(implications.values()))
        
        return implications
    
    def _calculate_measurement_support(
        self,
        measurements: Dict[str, float]
    ) -> float:
        """Calculate measurement support for necessity."""
        if not measurements:
            return 0.0
            
        # Calculate entropy reduction
        base_entropy = -np.log2(1.0 / len(measurements))
        actual_entropy = -np.sum(
            [p * np.log2(p) for p in measurements.values() if p > 0]
        )
        
        # Normalize to [0, 1]
        support = 1.0 - (actual_entropy / base_entropy)
        return max(0.0, min(1.0, support))
    
    def _calculate_evolution_support(
        self,
        evolution_data: Dict[str, Any]
    ) -> float:
        """Calculate evolution support for necessity."""
        states = evolution_data.get('states', [])
        if len(states) < 2:
            return 0.0
            
        # Calculate state vector changes
        changes = [
            np.linalg.norm(states[i+1] - states[i])
            for i in range(len(states)-1)
        ]
        
        # Calculate stability measure
        stability = 1.0 / (1.0 + np.std(changes))
        return stability
    
    def _calculate_proof_strength(
        self,
        analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall proof strength."""
        weights = {
            'quantum_evidence': 0.6,
            'modal_implications': 0.4
        }
        
        strength = 0.0
        for key, weight in weights.items():
            if key in analysis:
                component = analysis[key]
                if isinstance(component, dict):
                    # Use the overall strength metric for each component
                    if 'evidence_strength' in component:
                        strength += weight * component['evidence_strength']
                    elif 'necessity_strength' in component:
                        strength += weight * component['necessity_strength']
        
        return strength

def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of all results.
    
    Args:
        results: Dictionary containing all analysis results
        
    Returns:
        Dictionary of comprehensive analysis
    """
    analysis = {}
    
    # Analyze quantum results
    result_analyzer = ResultAnalyzer()
    analysis['quantum_analysis'] = result_analyzer.analyze_quantum_results(
        results.get('quantum_results', {})
    )
    
    # Analyze necessity proof
    necessity_analyzer = NecessityAnalyzer()
    analysis['proof_analysis'] = necessity_analyzer.analyze_necessity_proof(
        results.get('quantum_results', {}),
        results.get('modal_results', {})
    )
    
    return analysis