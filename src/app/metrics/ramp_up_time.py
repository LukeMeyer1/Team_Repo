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

@register("ramp_up_time")
class RampUpTimeMetric(BaseMetric):
    name = "ramp_up_time"

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
            except Exception:
                readme = ""

        if "usage" in readme or "example" in readme:
            score = 0.6
            if "```" in readme:  # presence of code block
                score += 0.3
            return min(score, 1.0)
        return 0.2