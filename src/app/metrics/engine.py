"""
Metrics orchestration and scoring engine.

This module handles the parallel execution of metrics and assembly of
results into NDJSON format. It provides the core orchestration logic
that coordinates multiple metrics and produces the final NetScore.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter_ns
import json
from typing import Dict, Any, List
from .base import MetricResult, ResourceBundle, Metric

def _extract_model_name(model_url: str) -> str:
    """
    Extract model name from HuggingFace URL.

    Args:
        model_url: Full model URL

    Returns:
        Model name without organization prefix (e.g., "bert-base-uncased")
    """
    if not model_url:
        return "unknown"

    # Handle various HuggingFace URL formats
    # https://huggingface.co/google-bert/bert-base-uncased -> bert-base-uncased
    # https://huggingface.co/bert-base-uncased -> bert-base-uncased
    # https://huggingface.co/openai/whisper-tiny/tree/main -> whisper-tiny
    # bert-base-uncased -> bert-base-uncased

    if "huggingface.co" in model_url:
        # Remove trailing slash and split
        url_path = model_url.rstrip('/').split('huggingface.co/')[-1]
        path_parts = url_path.split('/')

        # Handle URLs with /tree/, /blob/, /raw/ etc.
        if len(path_parts) >= 2:
            # Check if there's a /tree/, /blob/, etc. in the path
            if len(path_parts) >= 3 and path_parts[2] in ['tree', 'blob', 'raw', 'resolve']:
                # Format: org/model/tree/branch -> return model
                return path_parts[1]
            else:
                # Format: org/model -> return model
                return path_parts[1] if len(path_parts) > 1 else path_parts[0]

    # Direct model name or fallback
    model_name = model_url.split('/')[-1] if '/' in model_url else model_url
    return model_name

def _format_size_score(result: MetricResult) -> Dict[str, float]:
    """
    Format size score as device compatibility object.

    Args:
        result: MetricResult from size_score metric

    Returns:
        Dict with device-specific scores or default structure if unavailable
    """
    # If the size metric computed device scores, they should be in the result
    # For now, create a default structure since the current size metric returns a single score
    if result.score == 0.0:
        # No size data available
        return {
            "raspberry_pi": 0.0,
            "jetson_nano": 0.0,
            "desktop_pc": 0.0,
            "aws_server": 0.0
        }
    else:
        # Map single score to device categories (this is a simplified mapping)
        # In a full implementation, the size metric should return device-specific scores
        score = result.score
        return {
            "raspberry_pi": round(max(0.0, score - 0.6), 2),  # Stricter for small devices
            "jetson_nano": round(max(0.0, score - 0.3), 2),
            "desktop_pc": round(score, 2),
            "aws_server": round(score, 2)
        }

def compute_net_score(results: Dict[str, MetricResult], weights_profile: Dict[str, float]) -> float:
    """
    Compute weighted NetScore from individual metric results.
    
    Implements the NetScore formula from the specification:
    NetScore = (Σ(weight_i × score_i)) / (Σ(weight_i))
    
    Args:
        results: Dictionary mapping metric names to their MetricResult objects
        weights_profile: Dictionary mapping metric names to their weights
        
    Returns:
        Weighted average score in range [0, 1], or 0.0 if no valid weights
        
    Future Enhancement Notes:
    - Add weight validation (ensure weights sum to reasonable value)
    - Support for metric dependencies (some metrics may depend on others)
    - Configurable score normalization strategies
    - Logging for weight mismatches or missing metrics
    """
    if not results or not weights_profile:
        return 0.0
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    # Apply weights to each metric that has both a result and a weight
    for metric_name, result in results.items():
        weight = weights_profile.get(metric_name, 0.0)
        if weight > 0:  # Only include metrics with positive weights
            weighted_sum += result.score * weight
            weight_sum += weight
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0

def assemble_ndjson_row(bundle: ResourceBundle, results: Dict[str, MetricResult],
                        net_score: float, latency_ms: int, category: str = "MODEL") -> str:
    """
    Assemble NDJSON output row for a resource bundle.
    
    Creates the final output format expected by the auto-grader, including
    the model URL, NetScore, timing information, and all individual metric scores.
    
    Args:
        bundle: Resource bundle that was evaluated
        results: Dictionary of metric results
        net_score: Computed weighted average score
        latency_ms: Total processing time for this bundle
        category: Resource category (MODEL, DATASET, CODE)
        
    Returns:
        JSON string in NDJSON format for output
        
    Future Enhancement Notes:
    - Add JSON schema validation
    - Support for additional metadata fields
    - Configurable precision for score rounding
    - Include metric notes in debug mode
    """
    # Extract model name from URL for the expected format
    model_name = _extract_model_name(bundle.model_url)

    # Start with required fields per specification (matching expected format exactly)
    ndjson_row = {
        "name": model_name,
        "category": category,  # Category determined by caller based on CSV position
        "net_score": round(net_score, 2),  # 2 decimal precision to match expected
        "net_score_latency": latency_ms
    }

    # Add individual metric scores and their latencies in specific order
    # Order matches the target format: each metric followed immediately by its latency
    metric_order = [
        "ramp_up_time",
        "bus_factor",
        "performance_claims",
        "license",
        "size_score",
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality"
    ]

    for metric_name in metric_order:
        if metric_name in results:
            result = results[metric_name]

            # Handle size_score specially - it should be an object with device scores
            if metric_name == "size_score":
                ndjson_row[metric_name] = _format_size_score(result)
            else:
                ndjson_row[metric_name] = round(result.score, 2)  # 2 decimal precision

            ndjson_row[f"{metric_name}_latency"] = result.latency_ms

    # Add any remaining metrics not in the predefined order (for completeness)
    for metric_name in sorted(results.keys()):
        result = results[metric_name]

        # Handle size_score specially - it should be an object with device scores
        if metric_name == "size_score":
            ndjson_row[metric_name] = _format_size_score(result)
        else:
            ndjson_row[metric_name] = round(result.score, 2)  # 2 decimal precision

        ndjson_row[f"{metric_name}_latency"] = result.latency_ms
    
    # Use compact JSON format to match expected layout
    return json.dumps(ndjson_row, separators=(',', ':'))

def run_bundle(bundle: ResourceBundle, metrics: List[Metric],
               weights_profile: Dict[str, float], category: str = "MODEL") -> str:
    """
    Execute all metrics for a resource bundle and return NDJSON result.
    
    This is the main orchestration function that:
    1. Runs all metrics in parallel using ThreadPoolExecutor
    2. Collects results and computes NetScore
    3. Assembles final NDJSON output string
    
    Args:
        bundle: Resource bundle to evaluate (model + datasets + code)
        metrics: List of metric instances to execute
        weights_profile: Weight configuration for NetScore calculation
        category: Resource category (MODEL, DATASET, CODE)
        
    Returns:
        NDJSON string ready for output
        
    Future Enhancement Notes:
    - Add timeout handling for slow metrics
    - Implement retry logic for transient failures  
    - Add progress reporting for long-running evaluations
    - Support for metric dependency ordering
    - Resource usage monitoring (memory, CPU)
    """
    t0 = perf_counter_ns()
    futures = {}
    
    # Determine optimal thread pool size
    # Cap at 8 to avoid overwhelming APIs, but respect CPU count
    max_workers = min(8, (os.cpu_count() or 4), len(metrics))
    
    # Execute all metrics in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit all metric computations
        for metric in metrics:
            future = ex.submit(metric.compute, bundle)
            futures[future] = metric.name
        
        # Collect results as they complete
        results = {}
        for fut in as_completed(futures):
            metric_name = futures[fut]
            try:
                metric_result = fut.result()
                results[metric_name] = metric_result
            except Exception as e:
                # TODO: Add proper logging here
                # For now, create a failed result to avoid breaking the pipeline
                results[metric_name] = MetricResult(
                    score=0.0,
                    latency_ms=0,
                    notes=f"Metric computation failed: {str(e)}"
                )
    
    # Compute final weighted score
    net_score = compute_net_score(results, weights_profile)
    
    # Total processing time for this bundle
    total_latency_ms = int((perf_counter_ns() - t0) / 1_000_000)
    
    return assemble_ndjson_row(bundle, results, net_score, total_latency_ms, category)
