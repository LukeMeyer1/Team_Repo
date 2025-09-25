import re
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

@register("ramp_up_time")
class RampUpTimeMetric(BaseMetric):
    name = "ramp_up_time"

    def _compute_score(self, resource: ResourceBundle) -> float:
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        readme = data.get("cardData", {}).get("README", "").lower()

        if "usage" in readme or "example" in readme:
            score = 0.6
            if "```" in readme:  # presence of code block
                score += 0.3
            return min(score, 1.0)
        return 0.2