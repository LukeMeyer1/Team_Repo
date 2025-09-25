"""
Model size compatibility metric for different deployment targets.

Evaluates model size compatibility across various hardware platforms
from resource-constrained devices to high-performance servers.
"""

import re
from typing import Optional
from pathlib import Path

from app.metrics.base import ResourceBundle, SizeScore
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import HuggingFace client with error handling
try:
    from app.integrations.huggingface_client import hf_client
    _HF_CLIENT_AVAILABLE = True
except ImportError:
    _HF_CLIENT_AVAILABLE = False
    hf_client = None


# Weight file patterns for identifying model weight files
_WEIGHT_PATTERNS = [
    r"\.safetensors$", r"\.bin$", r"\.gguf$",
    r"pytorch_model-.*\.bin$", r"model-\d+-of-\d+\.safetensors$",
    r"pytorch_model\.bin$", r"model\.safetensors$",
    r"model\.safetensors\.index\.json$",      # ST sharded index
    r"pytorch_model\.bin\.index\.json$",      # BIN sharded index
]


def _is_weight_file(path: str) -> bool:
    """Check if a file path matches weight file patterns."""
    return any(re.search(p, path) for p in _WEIGHT_PATTERNS)


def _extract_model_id_from_url(model_url: str) -> Optional[str]:
    """Extract model ID from HuggingFace URL."""
    # Handle various HuggingFace URL formats
    # https://huggingface.co/bert-base-uncased
    # https://huggingface.co/microsoft/DialoGPT-medium
    # https://huggingface.co/models/bert-base-uncased

    if "huggingface.co" not in model_url:
        return None

    # Remove trailing slashes and query parameters
    url_clean = model_url.rstrip('/').split('?')[0]

    # Extract path after huggingface.co
    parts = url_clean.split('/')

    try:
        # Find the index of the part containing huggingface.co
        hf_index = -1
        for i, part in enumerate(parts):
            if 'huggingface.co' in part:
                hf_index = i
                break

        if hf_index == -1:
            return None

        path_parts = parts[hf_index + 1:]

        # Skip 'models' if present
        if path_parts and path_parts[0] == 'models':
            path_parts = path_parts[1:]

        # Should have at least model-name (can be single part for default namespace)
        if len(path_parts) >= 1:
            # For single part, return as-is (default namespace)
            # For multiple parts, return username/model-name
            if len(path_parts) == 1:
                return path_parts[0]
            else:
                return '/'.join(path_parts[:2])

    except (ValueError, IndexError):
        pass

    return None


@register("size_score")
class SizeScoreMetric(BaseMetric):
    """
    Evaluates model size compatibility across deployment targets.

    This metric downloads models locally to calculate actual weight file sizes,
    then computes device-specific compatibility scores for:
    - Raspberry Pi: Resource-constrained edge devices
    - Jetson Nano: AI-focused edge computing platforms
    - Desktop PC: Consumer hardware with moderate resources
    - AWS Server: Cloud instances with high-end resources

    Size Compatibility Ranges:
    - Raspberry Pi: < 100MB = 1.0, 100-500MB = linear decay to 0.0
    - Jetson Nano: < 1GB = 1.0, 1-4GB = linear decay to 0.0
    - Desktop PC: < 5GB = 1.0, 5-20GB = linear decay to 0.0
    - AWS Server: < 50GB = 1.0, 50-200GB = linear decay to 0.0

    Scoring Approach:
    Overall score = average of device compatibility scores
    """

    name = "size_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate model size compatibility across deployment targets.
        Downloads the model locally and calculates actual size from weight files.
        """
        if not _HF_CLIENT_AVAILABLE:
            return 0.0  # Cannot analyze without HuggingFace client

        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is None:
            return 0.0

        device_scores = self._compute_device_scores(model_size_mb)
        return sum(device_scores.values()) / len(device_scores)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        if not _HF_CLIENT_AVAILABLE:
            return f"HuggingFace client not available. Cannot analyze model size for {resource.model_url}."

        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is not None:
            return f"Model size: {model_size_mb:.1f}MB. Device compatibility computed from downloaded model weight files."
        return f"Could not fetch model size from {resource.model_url}. Score set to 0.0 (unevaluable)."

    def _fetch_model_size(self, model_url: str) -> Optional[float]:
        """
        Fetch total weight file size (in MB) by downloading the model locally.

        Strategy:
          - Extract model ID from URL
          - Download model using HuggingFace client
          - Calculate size of weight files only
          - Clean up downloaded files
        """
        if not _HF_CLIENT_AVAILABLE:
            return None

        try:
            model_id = _extract_model_id_from_url(model_url)
            if not model_id:
                return None

            # Download model to a temporary cache directory
            cache_dir = f"models/size_metric_cache/{model_id.replace('/', '_')}"
            download_result = hf_client.download_model(
                model_id=model_id,
                cache_dir=cache_dir,
                ignore_patterns=["*.h5", "*.md", "*.txt"]  # Skip non-weight files
            )

            if not download_result['success']:
                return None

            # Calculate size of weight files
            local_path = Path(download_result['local_path'])
            total_bytes = 0

            if local_path.exists():
                for file_path in local_path.rglob('*'):
                    if file_path.is_file() and _is_weight_file(file_path.name):
                        total_bytes += file_path.stat().st_size

            # Clean up downloaded files to save space
            try:
                import shutil
                if local_path.exists():
                    shutil.rmtree(local_path.parent)
            except Exception:
                pass  # Cleanup failure shouldn't affect the metric

            return (total_bytes / (1024 * 1024)) if total_bytes > 0 else None

        except Exception:
            return None

    def _compute_device_scores(self, model_size_mb: float) -> SizeScore:
        """
        Compute device-specific compatibility scores based on model size.

        Args:
            model_size_mb: Model size in megabytes

        Returns:
            SizeScore with compatibility scores for each device type
        """
        thresholds = {
            "raspberry_pi": {"optimal": 100, "max": 500},
            "jetson_nano": {"optimal": 1000, "max": 4000},
            "desktop_pc": {"optimal": 5000, "max": 20000},
            "aws_server": {"optimal": 50000, "max": 200000}
        }

        scores: SizeScore = {}
        for device, limits in thresholds.items():
            if model_size_mb <= limits["optimal"]:
                scores[device] = 1.0
            elif model_size_mb <= limits["max"]:
                # Linear decay from optimal to max threshold
                ratio = (model_size_mb - limits["optimal"]) / (limits["max"] - limits["optimal"])
                scores[device] = max(0.0, 1.0 - ratio)
            else:
                scores[device] = 0.0

        return scores
