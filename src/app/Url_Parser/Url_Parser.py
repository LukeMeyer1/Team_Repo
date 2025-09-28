"""
Utilities for parsing Hugging Face model URLs and fetching metadata.

This module centralizes URL parsing and API requests so metric implementations
don't need to duplicate boilerplate logic.
"""

import requests
from urllib.parse import urlparse

# Base template for Hugging Face model API
HF_API_MODEL = "https://huggingface.co/api/models/{model_id}"


def parse_model_id(model_url: str) -> str:
    """
    Extract the model ID from a Hugging Face model URL.

    Args:
        model_url: Full model URL, e.g. https://huggingface.co/bert-base-uncased

    Returns:
        Model ID string, e.g. 'bert-base-uncased'
    """
    parsed = urlparse(model_url)
    if "huggingface.co" not in parsed.netloc:
        raise ValueError(f"Not a valid Hugging Face model URL: {model_url}")

    return parsed.path.strip("/")
    

def fetch_model_metadata(model_url: str) -> dict:
    """
    Fetch metadata for a Hugging Face model via the REST API.

    Args:
        model_url: Full model URL (e.g., https://huggingface.co/bert-base-uncased)

    Returns:
        Dictionary with model metadata (JSON response from HF API)
    """
    model_id = parse_model_id(model_url)
    api_url = HF_API_MODEL.format(model_id=model_id)

    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()
    return resp.json()
