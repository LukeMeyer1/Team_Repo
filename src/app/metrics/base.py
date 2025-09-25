"""
Core data structures and interfaces for the metrics system.

This module defines the fundamental types and protocols that all metrics
implement, providing type safety and consistent interfaces across the system.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypedDict, Optional, List

class SizeScore(TypedDict):
    """
    Device-specific size compatibility scores.
    
    Used by size_score metric to evaluate model compatibility across
    different deployment targets. Each score should be in [0, 1] range.
    
    Future Implementation Notes:
    - raspberry_pi: Models < 100MB get 1.0, linear decay to 500MB = 0.0
    - jetson_nano: Models < 1GB get 1.0, linear decay to 4GB = 0.0  
    - desktop_pc: Models < 5GB get 1.0, linear decay to 20GB = 0.0
    - aws_server: Models < 50GB get 1.0, linear decay to 200GB = 0.0
    """
    raspberry_pi: float
    jetson_nano: float 
    desktop_pc: float
    aws_server: float

@dataclass
class MetricResult:
    """
    Result from a single metric computation.
    
    Standardized output format ensuring all metrics provide consistent
    data structure with score, timing, and optional diagnostic information.
    
    Attributes:
        score: Metric value in range [0, 1] where 1 is best
        latency_ms: Computation time in milliseconds for performance tracking
        notes: Optional human-readable description of computation method or issues
    """
    score: float           # Range [0, 1], higher is better
    latency_ms: int       # Computation time for performance monitoring
    notes: Optional[str] = None  # Diagnostic info or computation method description

@dataclass 
class ResourceBundle:
    """
    Collection of related URLs for comprehensive model evaluation.
    
    Groups model with its associated datasets and code repositories to enable
    holistic scoring that considers the complete ecosystem around a model.
    
    Future Enhancement Notes:
    - Add URL validation and parsing
    - Include metadata like model type, framework, etc.
    - Support for multiple model variants (different quantizations)
    - Cache repository metadata to avoid repeated API calls
    
    Attributes:
        model_url: Primary Hugging Face model URL to evaluate
        dataset_urls: Associated dataset URLs (for dataset_quality metric)
        code_urls: Related code repository URLs (for code_quality metric)
    """
    model_url: str           # Primary model URL from Hugging Face
    dataset_urls: List[str]  # Associated datasets for quality evaluation
    code_urls: List[str]     # Related repositories for code quality analysis
    model_id: Optional[str] = None            # Model id for metric uses

class Metric(Protocol):
    """
    Protocol defining the interface all metrics must implement.
    
    Ensures type safety and consistent behavior across all metric implementations.
    All metrics must provide a unique name and implement the compute method.
    
    The protocol allows for both class-based metrics (with __init__) and
    function-based metrics, as long as they satisfy this interface.
    """
    name: str  # Unique identifier for metric registration and weight assignment
    
    def compute(self, resource: ResourceBundle) -> MetricResult:
        """
        Compute metric score for given resource bundle.
        
        Args:
            resource: Bundle containing model URL and related resources
            
        Returns:
            MetricResult with score in [0, 1], timing, and optional notes
            
        Raises:
            Should handle errors gracefully and return meaningful MetricResult
            rather than propagating exceptions to the orchestrator.
        """
        ...
