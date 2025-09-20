"""
Wrapper for Purdue GenAI Studio.

Docs: https://www.rcac.purdue.edu/knowledge/genaistudio/api
Endpoint: https://genai.rcac.purdue.edu/api/chat/completions

How to use this:
------------------------------
from integrations.genai_studio import GenAIStudio

client = GenAIStudio.from_env()  # reads GENAI_API_KEY
text = "Paste the thing you want evaluated..."
notes = client.evaluate(
    method=(
        "Evaluate for correctness, clarity, and tone. "
        "Return concise bullet points with 'Good', 'Issues', 'Suggestions'."
    ),
    contents=text,
    model="llama3.1:latest",
)
print(notes)

"""
from __future__ import annotations

import os
from typing import Dict, List, Optional
from pathlib import Path

import requests


def _load_env_file():
    """Load environment variables from .env file in integrations folder."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load environment variables on import
_load_env_file()

__all__ = ["GenAIStudio"]


class GenAIStudio:
    """
    Primary entrypoint: :meth:`evaluate`.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
        timeout: float = 40.0,
    ) -> None:
        # Use environment variable if no api_key provided
        if api_key is None:
            api_key = os.getenv("GENAI_API_KEY")

        if not api_key:
            raise ValueError("api_key is required. Set GENAI_API_KEY or pass explicitly.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls, env_var: str = "GENAI_API_KEY", **kwargs) -> "GenAIStudio":
        return cls(os.getenv(env_var), **kwargs)

    # ---------------- public API ----------------
    def evaluate(
        self,
        method: str,
        contents: str,
        *,
        model: str = "llama3.1:latest",
        temperature: Optional[float] = 0.2,
        max_tokens: Optional[int] = None,
        system_preamble: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """Apply the given *method of evaluation* to *contents* and return text.

        Parameters
        ----------
        method : str
            Short instruction describing how to evaluate (rubric, criteria, style).
        contents : str
            The text or data to evaluate.
        model : str, default "llama3.1:latest"
            Model name supported by GenAI Studio.
        temperature : float | None, default 0.2
            Mildly deterministic by default.
        max_tokens : int | None
            Optional cap for output length.
        system_preamble : str | None
            Optional system message prefix. A sensible default is provided.
        extra : dict | None
            Extra fields passed through to the API body (provider-specific).
        """
        sys_msg = system_preamble or (
            "You are a precise evaluator. Read the user's instructions for the"
            " evaluation method, then assess the provided contents accordingly."
            " Prefer concise, actionable output."
        )

        # Encode the calling contract explicitly so prompts stay stable.
        user_prompt = (
            "# Evaluation Method\n" + method.strip() +
            "\n\n# Contents To Evaluate\n" + contents.strip()
        )

        body: Dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if extra:
            body.update(extra)

        r = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        _raise_for_status(r)
        data = r.json()
        return _extract_content(data)


# ---------------- helpers ----------------

def _raise_for_status(r: requests.Response) -> None:
    if 200 <= r.status_code < 300:
        return
    try:
        err = r.json()
    except Exception:
        err = {"error": r.text}
    raise requests.HTTPError(f"HTTP {r.status_code}: {err}")


def _extract_content(data: Dict) -> str:
    try:
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        return ""
