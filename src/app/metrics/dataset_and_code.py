import re
from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric
from app.Url_Parser.Url_Parser import *

@register("dataset_and_code_score")
class DatasetAndCodeScoreMetric(BaseMetric):
    """
    Evaluates presence of dataset and code references in model documentation.
    """

    name = "dataset_and_code_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate presence of dataset and code references through simple checks.
        """
        r = requests.get(HF_API_MODEL.format(model_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        readme = data.get("cardData", {}).get("README", "").lower()

        # Base score
        score = 0.2

        # Check for dataset mentions
        dataset_keywords = ["dataset", "training data", "evaluation data", "benchmark", "corpus"]
        has_dataset_mention = any(keyword in readme for keyword in dataset_keywords)
        if has_dataset_mention:
            score += 0.3

        # Check for code/repository links
        code_patterns = ["github.com", "gitlab.com", "code", "repository", "repo"]
        has_code_reference = any(pattern in readme for pattern in code_patterns)
        if has_code_reference:
            score += 0.3

        # Check for URLs provided in resource bundle
        if resource.dataset_urls:
            score += 0.1
        if resource.code_urls:
            score += 0.1

        return min(score, 1.0)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return f"Dataset and code analysis for {resource.model_url} based on README content and provided URLs."