import re
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

@register("license")
class LicenseMetric(BaseMetric):
    name = "license"

    def _compute_score(self, resource: ResourceBundle) -> float:
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()

        license_tag = (data.get("cardData", {}).get("license") or "").lower()
        readme = data.get("cardData", {}).get("README", "").lower()

        # LGPLv2.1 is required
        if "lgpl-2.1" in license_tag or "lgpl v2.1" in readme:
            return 1.0
        elif license_tag in ["mit", "apache-2.0", "bsd-3-clause"]:
            return 0.7
        elif license_tag:
            return 0.3
        return 0.0