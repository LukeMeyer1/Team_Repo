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

@register("performance_claims")
class PerformanceClaimsMetric(BaseMetric):
    name = "performance_claims"

    def _compute_score(self, resource: ResourceBundle) -> float:
        if not _HF_CLIENT_AVAILABLE or not hf_client:
            return 0.2

        # Get README content using HuggingFace client
        readme = hf_client.get_model_readme(resource.model_id)
        if not readme:
            return 0.2

        readme_lower = readme.lower()

        # Look for evaluation metrics
        metrics = ["bleu", "f1", "accuracy", "rouge", "perplexity", "cer", "wer"]
        found = sum(1 for m in metrics if m in readme_lower)

        # Look for benchmark tables
        has_table = " | " in readme_lower and "---" in readme_lower  # crude Markdown table check

        # Look for citations/links
        has_citation = bool(re.search(r"(arxiv\.org|paperswithcode\.com|huggingface\.co/evaluate)", readme_lower))

        # Scoring
        score = 0.2
        if found:
            score += 0.1 * min(found, 5)  # up to 0.5
        if has_table:
            score += 0.2
        if has_citation:
            score += 0.2
        return min(score, 1.0)