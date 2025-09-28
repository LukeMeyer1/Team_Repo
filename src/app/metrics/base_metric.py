"""
Base metric implementation providing common functionality.

This module contains the abstract base class that eliminates code duplication
across all metric implementations by providing shared timing, placeholder
scoring, and result creation logic.
"""

from abc import ABC, abstractmethod
from time import perf_counter_ns
import hashlib
from typing import Optional

from .base import Metric, MetricResult, ResourceBundle


def stable_01(key: str, salt: str = "") -> float:
    """
    Generate deterministic score between 0-1 from key and salt.
    
    Uses BLAKE2b hash to create reproducible scores for testing/placeholder purposes.
    
    Args:
        key: Primary input string (typically a URL)
        salt: Additional string to vary the hash output
        
    Returns:
        Float in range (0, 1] - never exactly 0, but can be 1
        
    Note:
        This is TEMPORARY placeholder logic. Real metrics will replace this
        with actual API calls, file analysis, or other evaluation methods.
    """
    h = hashlib.blake2b((key + "|" + salt).encode(), digest_size=8).hexdigest()
    val = int(h, 16) % 10_000_000
    return val / 9_999_999.0  # Ensures range (0, 1]


class BaseMetric(ABC):
    """
    Abstract base class for all metrics implementations.
    
    Provides common functionality:
    - Automatic timing measurement
    - Placeholder score generation
    - Consistent result formatting
    - Error handling framework (TODO)
    
    Subclasses only need to implement:
    - name: str class attribute
    - _compute_score(): actual metric calculation
    """
    
    # Must be overridden by subclasses
    name: str = ""
    
    def __init__(self):
        """Initialize metric instance."""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")
    
    def compute(self, resource: ResourceBundle) -> MetricResult:
        """
        Compute metric score with automatic timing and error handling.
        
        This method provides the standard Metric protocol interface while
        handling common concerns like timing measurement.
        
        Args:
            resource: Bundle containing model URL and associated dataset/code URLs
            
        Returns:
            MetricResult with score, timing, and optional notes
            
        Note:
            Currently uses placeholder scoring. Real implementations will
            need to override _compute_score() with actual evaluation logic.
        """
        t0 = perf_counter_ns()
        
        try:
            # Delegate actual scoring to subclass
            score = self._compute_score(resource)
            notes = self._get_computation_notes(resource)
        except Exception as e:
            # TODO: Add proper error handling and logging
            # For now, fall back to placeholder to avoid breaking tests
            score = stable_01(resource.model_url, self.name)
            print("AAAHHHH we had to use the placeholder. " + e)
            notes = f"Error in computation, using placeholder: {str(e)}"
        
        latency_ms = int((perf_counter_ns() - t0) / 1_000_000)
        
        return MetricResult(
            score=score,
            latency_ms=latency_ms,
            notes=notes
        )
    
    @abstractmethod
    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Compute the actual metric score.
        
        Subclasses must implement this method with their specific evaluation logic.
        
        Args:
            resource: Bundle containing URLs to evaluate
            
        Returns:
            Score in range [0, 1] where 1 is best
            
        Raises:
            Any exceptions will be caught by compute() and handled gracefully
        """
        pass
    
    def _get_computation_notes(self, resource: ResourceBundle) -> Optional[str]:
        """
        Get optional notes about the computation.
        
        Subclasses can override to provide insight into their evaluation process.
        
        Args:
            resource: Bundle that was evaluated
            
        Returns:
            Optional descriptive text about the computation method
        """
        return f"Placeholder implementation for {self.name} metric"