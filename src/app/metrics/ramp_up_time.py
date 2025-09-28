import re
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register

# Import HuggingFace client
try:
    from app.integrations.huggingface_client import hf_client
    _HF_CLIENT_AVAILABLE = True
except ImportError:
    _HF_CLIENT_AVAILABLE = False
    hf_client = None

@register("ramp_up_time")
class RampUpTimeMetric(BaseMetric):
    name = "ramp_up_time"

    def _compute_score(self, resource: ResourceBundle) -> float:
        if not _HF_CLIENT_AVAILABLE or not hf_client:
            return 0.2

        # Get README content using HuggingFace client
        readme = hf_client.get_model_readme(resource.model_id)
        if not readme:
            return 0.2

        readme_lower = readme.lower()

        if "usage" in readme_lower or "example" in readme_lower:
            score = 0.6
            if "```" in readme_lower:  # presence of code block
                score += 0.3
            return min(score, 1.0)
        return 0.2