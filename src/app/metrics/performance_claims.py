import re
import requests
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

# Import HuggingFace client for README access
try:
    from app.integrations.huggingface_client import hf_client
except ImportError:
    hf_client = None

@register("performance_claims")
class PerformanceClaimsMetric(BaseMetric):
    name = "performance_claims"

    def _compute_score(self, resource: ResourceBundle) -> float:
        model_id = parse_model_id(resource.model_url)

        # Get README content using hf_client for more reliable access
        readme = ""
        if hf_client:
            readme_content = hf_client.get_model_readme(model_id)
            if readme_content:
                readme = readme_content.lower()


        # Fallback to direct README fetching
        if not readme:
            try:
                # Try the direct README.md file from the repo
                readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                r = requests.get(readme_url, timeout=10)
                if r.status_code == 200:
                    readme = r.text.lower()
                else:
                    # Try alternative path
                    readme_url2 = f"https://huggingface.co/{model_id}/resolve/main/README.md"
                    r2 = requests.get(readme_url2, timeout=10)
                    if r2.status_code == 200:
                        readme = r2.text.lower()


            except Exception as e:
                readme = ""

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