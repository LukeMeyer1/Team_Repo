import re
from datetime import datetime, timedelta
from .base_metric import BaseMetric
from .base import ResourceBundle
from .registry import register
from ..Url_Parser.Url_Parser import *

@register("bus_factor")
class BusFactorMetric(BaseMetric):
    name = "bus_factor"

    def _compute_score(self, resource: ResourceBundle) -> float:
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()

        author = data.get("author", "")
        org = data.get("organization")
        repo_url = None

        # Try to find external repo in "siblings" or "cardData"
        for sibling in data.get("siblings", []):
            if sibling.get("rfilename", "").endswith(".git"):
                repo_url = sibling["rfilename"]
        repo_url = repo_url or data.get("cardData", {}).get("repository")

        if not repo_url or "github.com" not in repo_url:
            return 0.3 if org else 0.2

        # Extract owner/repo
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return 0.4
        owner, repo = match.groups()

        # GitHub contributors API
        gh_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
        contributors = requests.get(gh_url, timeout=10).json()
        num_contributors = len(contributors) if isinstance(contributors, list) else 0

        # GitHub commits API (recent activity)
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        commits = requests.get(commits_url, timeout=10, params={"per_page": 5}).json()
        recent_commit = None
        if isinstance(commits, list) and commits:
            recent_commit = commits[0]["commit"]["author"]["date"]
            recent_commit_date = datetime.fromisoformat(recent_commit.replace("Z", "+00:00"))
            recency_days = (datetime.utcnow() - recent_commit_date).days
        else:
            recency_days = 9999

        # Score formula
        score = 0.2
        if num_contributors > 5:
            score += 0.4
        elif num_contributors > 1:
            score += 0.2

        if recency_days < 30:
            score += 0.4
        elif recency_days < 180:
            score += 0.2

        if org:
            score += 0.1

        return min(score, 1.0)