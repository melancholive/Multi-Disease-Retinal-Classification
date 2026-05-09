import json
import os
from threading import Lock

import numpy as np
import torch
import httpx
from PIL import Image
from transformers import pipeline

from app.utils.json_parsing import extract_json


_MEDGEMMA_PIPE = None
_MEDGEMMA_LOCK = Lock()


def _get_remote_medgemma_url() -> str | None:
    url = os.getenv("MEDGEMMA_API_URL")
    return url.strip() if url else None


def _get_remote_timeout() -> float:
    return float(os.getenv("MEDGEMMA_API_TIMEOUT", "600"))


def _get_medgemma_pipeline_kwargs() -> dict:
    """Use explicit placement by default to avoid Accelerate meta tensors."""

    device_map = os.getenv("MEDGEMMA_DEVICE_MAP")
    if device_map:
        return {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map,
        }

    if torch.cuda.is_available():
        return {
            "torch_dtype": torch.bfloat16,
            "device": 0,
        }

    return {
        "torch_dtype": torch.float32,
        "device": -1,
    }


def get_medgemma_pipe():
    """Lazily create and cache the local MedGemma pipeline.

    Loading MedGemma is expensive; do it on first use (not at import time).
    """

    global _MEDGEMMA_PIPE
    if _MEDGEMMA_PIPE is None:
        with _MEDGEMMA_LOCK:
            if _MEDGEMMA_PIPE is None:
                _MEDGEMMA_PIPE = pipeline(
                    task="image-text-to-text",
                    model="google/medgemma-4b-it",
                    **_get_medgemma_pipeline_kwargs(),
                )
    return _MEDGEMMA_PIPE


def _extract_remote_text(data) -> str:
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ("text", "answer", "generated_text", "response", "output"):
            value = data.get(key)
            if isinstance(value, str):
                return value

        return json.dumps(data)

    return str(data)


def call_remote_medgemma(
    api_url: str, prompt: str, max_new_tokens: int
) -> str:
    response = httpx.post(
        api_url,
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        },
        timeout=_get_remote_timeout(),
    )
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError:
        data = response.text

    return _extract_remote_text(data)


def medgemma_model(prompt: str, max_new_tokens: int = 2048) -> str:
    strict_prompt = f"""
You must return ONLY valid JSON.
No markdown.
No explanations outside JSON.
No code fences.

{prompt}
"""

    remote_url = _get_remote_medgemma_url()
    if remote_url:
        text = call_remote_medgemma(
            api_url=remote_url,
            prompt=strict_prompt,
            max_new_tokens=max_new_tokens,
        )
        parsed = extract_json(text)
        return json.dumps(parsed, indent=2)

    medgemma_pipe = get_medgemma_pipe()
    output = medgemma_pipe(
        text=[
            {
                "role": "user",
                "content": [{"type": "text", "text": strict_prompt}],
            }
        ],
        max_new_tokens=max_new_tokens,
    )

    generated = output[0].get("generated_text", output[0])

    if isinstance(generated, list):
        text = generated[-1].get("content", str(generated[-1]))
    else:
        text = str(generated)

    parsed = extract_json(text)

    return json.dumps(parsed, indent=2)
