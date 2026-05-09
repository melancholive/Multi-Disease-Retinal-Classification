import json

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from app.utils.json_parsing import extract_json


_MEDGEMMA_PIPE = None


def get_medgemma_pipe():
    """Lazily create and cache the local MedGemma pipeline.

    Loading MedGemma is expensive; do it on first use (not at import time).
    """

    global _MEDGEMMA_PIPE
    if _MEDGEMMA_PIPE is None:
        _MEDGEMMA_PIPE = pipeline(
            task="image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    return _MEDGEMMA_PIPE


def call_medgemma_text(
    medgemma_model,
    prompt,
    image=None,
    max_new_tokens=1024,
    do_sample=False,
):
    """
    Call MedGemma with text-only or image+text input.

    Parameters
    ----------
    medgemma_model : HuggingFace pipeline
        image-text-to-text pipeline

    prompt : str
        Prompt text

    image : optional
        PIL image, numpy array, tensor, or image path

    Returns
    -------
    str
        Extracted text response from MedGemma
    """

    messages = [{"role": "user", "content": []}]

    # optional image input
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            image = Image.fromarray(image.astype(np.uint8))

        elif isinstance(image, torch.Tensor):
            img_array = image.detach().cpu().numpy()

            if img_array.ndim == 3 and img_array.shape[0] in [1, 3]:
                img_array = np.transpose(img_array, (1, 2, 0))

            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)

            image = Image.fromarray(img_array.astype(np.uint8))

        messages[0]["content"].append({"type": "image", "image": image})

    # add text prompt
    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        output = medgemma_model(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # pipeline output handling
        if isinstance(output, str):
            return output

        if isinstance(output, list) and len(output) > 0:
            first = output[0]

            # common HF pipeline structure
            if isinstance(first, dict):
                generated = first.get("generated_text") or first.get("text") or first

                # conversational output
                if isinstance(generated, list):
                    last_message = generated[-1]

                    if isinstance(last_message, dict):
                        return last_message.get("content", str(last_message))

                    return str(last_message)

                return str(generated)

            return str(first)

        return str(output)

    except Exception as e:
        return json.dumps({"error": f"MedGemma inference failed: {str(e)}"})


def medgemma_model(prompt: str, max_new_tokens: int = 2048) -> str:
    strict_prompt = f"""
You must return ONLY valid JSON.
No markdown.
No explanations outside JSON.
No code fences.

{prompt}
"""

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
