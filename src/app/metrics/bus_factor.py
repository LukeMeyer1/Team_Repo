import re
import requests
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

# Import HuggingFace client for better model info access
try:
    from app.integrations.huggingface_client import hf_client
except ImportError:
    hf_client = None

@register("bus_factor")
class BusFactorMetric(BaseMetric):
    name = "bus_factor"

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        model_id = parse_model_id(resource.model_url)
        notes = f"Bus factor analysis for {model_id} based on author reputation, community engagement (downloads/likes), and maintenance activity."

        # Add specific metrics if available
        if hf_client:
            info_result = hf_client.get_model_info(model_id)
            if info_result and 'model_info' in info_result:
                model_info = info_result['model_info']
                downloads = getattr(model_info, 'downloads', 0) or 0
                likes = getattr(model_info, 'likes', 0) or 0
                author = getattr(model_info, 'author', '')

                metrics = []
                if author:
                    metrics.append(f"author: {author}")
                if downloads > 0:
                    metrics.append(f"downloads: {downloads:,}")
                if likes > 0:
                    metrics.append(f"likes: {likes}")

                if metrics:
                    notes += f" Metrics: {', '.join(metrics)}."

        return notes

    def _compute_score(self, resource: ResourceBundle) -> float:
        model_id = parse_model_id(resource.model_url)

        # Get model info using hf_client for better access
        model_info = None
        if hf_client:
            info_result = hf_client.get_model_info(model_id)
            if info_result and 'model_info' in info_result:
                model_info = info_result['model_info']

        # Fallback to direct API call
        if not model_info:
            try:
                r = requests.get(HF_API_MODEL.format(model_id=model_id), timeout=10)
                r.raise_for_status()
                data = r.json()
            except Exception:
                return 0.1  # Minimal score if we can't get any data
        else:
            # Convert model_info object to dict-like access
            data = {
                'author': getattr(model_info, 'author', None),
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'lastModified': getattr(model_info, 'lastModified', None),
                'createdAt': getattr(model_info, 'createdAt', None),
                'siblings': getattr(model_info, 'siblings', []),
                'cardData': getattr(model_info, 'cardData', {})
            }

        # Start with base score
        score = 0.1

        # Author/organization presence (indicates some level of maintenance)
        author = data.get("author", "")
        if author:
            # Well-known organizations get higher base score
            well_known_orgs = ['google', 'microsoft', 'facebook', 'openai', 'huggingface',
                              'anthropic', 'nvidia', 'salesforce', 'stanford', 'mit']
            if any(org in author.lower() for org in well_known_orgs):
                score += 0.4
            else:
                score += 0.2

        # Popularity metrics (downloads and likes indicate community usage)
        downloads = data.get('downloads', 0) or 0
        likes = data.get('likes', 0) or 0

        # Downloads scoring (log scale for wide range)
        if downloads >= 1000000:  # 1M+ downloads
            score += 0.3
        elif downloads >= 100000:  # 100K+ downloads
            score += 0.25
        elif downloads >= 10000:   # 10K+ downloads
            score += 0.2
        elif downloads >= 1000:    # 1K+ downloads
            score += 0.15
        elif downloads >= 100:     # 100+ downloads
            score += 0.1

        # Likes scoring (community engagement)
        if likes >= 1000:
            score += 0.2
        elif likes >= 100:
            score += 0.15
        elif likes >= 50:
            score += 0.1
        elif likes >= 10:
            score += 0.05

        # Recent activity (lastModified indicates maintenance)
        last_modified = data.get('lastModified')
        if last_modified:
            try:
                # Parse the lastModified date
                if isinstance(last_modified, str):
                    # Handle different date formats
                    if 'T' in last_modified:
                        modified_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    else:
                        modified_date = datetime.fromisoformat(last_modified)
                else:
                    modified_date = last_modified

                days_since_update = (datetime.now() - modified_date.replace(tzinfo=None)).days

                if days_since_update <= 30:      # Updated within a month
                    score += 0.2
                elif days_since_update <= 180:   # Updated within 6 months
                    score += 0.15
                elif days_since_update <= 365:   # Updated within a year
                    score += 0.1
            except Exception:
                pass  # Skip recency scoring if date parsing fails

        # Check for external GitHub repository (bonus points)
        repo_url = None
        for sibling in data.get("siblings", []):
            # Handle both dict and object cases
            if hasattr(sibling, 'rfilename'):
                rfilename = sibling.rfilename
            elif isinstance(sibling, dict):
                rfilename = sibling.get("rfilename", "")
            else:
                rfilename = ""

            if rfilename and rfilename.endswith(".git"):
                repo_url = rfilename

        cardData = data.get("cardData", {})
        if isinstance(cardData, dict):
            repo_url = repo_url or cardData.get("repository")

        if repo_url and "github.com" in repo_url:
            score += 0.1  # Bonus for having external repository

        return round(min(score, 1.0), 2)