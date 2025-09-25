import re
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

@register("performance_claims")
class PerformanceClaimsMetric(BaseMetric):
    name = "performance_claims"

    def _compute_score(self, resource: ResourceBundle) -> float:
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        readme = data.get("cardData", {}).get("README", "").lower()

        # Look for evaluation metrics
        metrics = ["bleu", "f1", "accuracy", "rouge", "perplexity", "cer", "wer"]
        found = sum(1 for m in metrics if m in readme)

        # Look for benchmark tables
        has_table = " | " in readme and "---" in readme  # crude Markdown table check

        # Look for citations/links
        has_citation = bool(re.search(r"(arxiv\.org|paperswithcode\.com|huggingface\.co/evaluate)", readme))

        # Scoring
        score = 0.2
        if found:
            score += 0.1 * min(found, 5)  # up to 0.5
        if has_table:
            score += 0.2
        if has_citation:
            score += 0.2
        return min(score, 1.0)