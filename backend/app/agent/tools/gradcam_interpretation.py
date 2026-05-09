import base64
import json
from io import BytesIO
from typing import Any

import numpy as np
import torch
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from app.utils.json_parsing import safe_parse_json


def _data_url_to_pil(data_url: str) -> Image.Image:
    header, b64 = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Unsupported data URL encoding")
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")



class GradCamInterpretationInput(BaseModel):
    gradcam_images: dict[str, Any] = Field(
        ...,
        description="""
        Dictionary of GradCAM images by model name.
        Example:
        {
          "efficientnet": "dataURL",
          "resnet": "dataURL",
          "shufflenet": "dataURL"
        }
        """,
    )

    top_disease: str = Field(..., description="Top predicted disease")

    per_class_f1: list[float] | None = None
    class_names: list[str] | None = None
    top_disease_index: int | None = None




def load_gradcam_image(image_input):
    if isinstance(image_input, str):
        if image_input.startswith("data:image/"):
            return _data_url_to_pil(image_input)
        return Image.open(image_input).convert("RGB")

    elif isinstance(image_input, np.ndarray):
        if image_input.max() <= 1.0:
            image_input = (image_input * 255).astype(np.uint8)

        return Image.fromarray(image_input)

    elif isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    elif isinstance(image_input, torch.Tensor):
        img_array = image_input.detach().cpu().numpy()

        if img_array.ndim == 3 and img_array.shape[0] in [1, 3]:
            img_array = np.transpose(img_array, (1, 2, 0))

        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array.astype(np.uint8))

    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def describe_location(center_x, center_y, image_shape):
    height, width = image_shape[0], image_shape[1]

    norm_x = center_x / width
    norm_y = center_y / height

    if 0.4 < norm_x < 0.6 and 0.4 < norm_y < 0.6:
        return "macula/fovea central retina"

    elif norm_x < 0.3:
        if norm_y < 0.3:
            return "superonasal retina"
        elif norm_y > 0.7:
            return "inferonasal retina"
        return "nasal retina / optic disc region"

    elif norm_x > 0.7:
        if norm_y < 0.3:
            return "superotemporal retina"
        elif norm_y > 0.7:
            return "inferotemporal retina"
        return "temporal retina / macular area"

    elif norm_y < 0.3:
        return "superior retina"

    elif norm_y > 0.7:
        return "inferior retina"

    return "mid peripheral retina"


def classify_attention_type(gradcam_array, high_attention_mask):
    high_attention_pixels = np.sum(high_attention_mask)

    if high_attention_pixels == 0:
        return "no clear attention pattern"

    y_coords, x_coords = np.where(high_attention_mask)

    if len(y_coords) < 10:
        return "minimal focused attention"

    variance_y = np.var(y_coords)
    variance_x = np.var(x_coords)

    total_variance = (variance_x + variance_y) / 2

    sorted_intensities = np.sort(gradcam_array[high_attention_mask])[::-1]

    top_20_sum = np.sum(sorted_intensities[: max(1, len(sorted_intensities) // 5)])

    total_sum = np.sum(sorted_intensities)

    concentration_ratio = top_20_sum / total_sum if total_sum > 0 else 0

    if concentration_ratio > 0.7:
        if total_variance < 3000:
            return "highly focused lesion-like attention"
        return "focused pathology cluster"

    elif concentration_ratio > 0.4:
        if total_variance < 5000:
            return "moderately focused pathology"
        return "diffuse pathology attention"

    return "broad distributed attention"


from scipy.ndimage import center_of_mass
from skimage.measure import label


def find_attention_peaks(gradcam_gray, high_attention_mask, num_peaks=3):
    labeled, num_features = label(high_attention_mask)

    peaks = []

    for i in range(1, min(num_features, num_peaks) + 1):
        component_mask = labeled == i

        if np.sum(component_mask) > 10:
            com_y, com_x = center_of_mass(component_mask)

            peak_intensity = np.max(gradcam_gray[component_mask])

            peaks.append(
                {
                    "peak_id": i,
                    "location": describe_location(com_x, com_y, gradcam_gray.shape),
                    "intensity": round(float(peak_intensity), 3),
                    "size_pixels": int(np.sum(component_mask)),
                }
            )

    return peaks


def analyze_attention_regions(gradcam_image, model_name, top_disease, f1_score=None):
    gradcam_array = np.array(gradcam_image)

    if len(gradcam_array.shape) == 3:
        gradcam_gray = np.mean(gradcam_array, axis=2)
    else:
        gradcam_gray = gradcam_array

    threshold = np.percentile(gradcam_gray, 80)

    high_attention_mask = gradcam_gray > threshold

    total_pixels = gradcam_gray.size
    high_attention_pixels = np.sum(high_attention_mask)

    attention_percentage = (high_attention_pixels / total_pixels) * 100

    if high_attention_pixels > 0:
        y_coords, x_coords = np.where(high_attention_mask)

        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        attention_location = describe_location(center_x, center_y, gradcam_gray.shape)

        peaks = find_attention_peaks(gradcam_gray, high_attention_mask)

    else:
        attention_location = "diffuse or uncertain attention"
        peaks = []

    return {
        "attention_percentage": round(float(attention_percentage), 2),
        "attention_location": attention_location,
        "mean_attention_intensity": round(float(np.mean(gradcam_gray)), 3),
        "max_attention_intensity": round(float(np.max(gradcam_gray)), 3),
        "model_performance_f1": (round(float(f1_score), 3) if f1_score else "N/A"),
        "attention_focus": classify_attention_type(gradcam_gray, high_attention_mask),
        "attention_peaks": peaks,
    }


def get_model_f1_for_disease(per_class_f1, class_names, disease_name):
    """
    Return the F1 score corresponding to a disease name.
    """

    try:
        disease_lower = disease_name.lower()

        for idx, class_name in enumerate(class_names):
            class_lower = class_name.lower()

            if (
                disease_lower == class_lower
                or disease_lower in class_lower
                or class_lower in disease_lower
            ):
                if idx < len(per_class_f1):
                    return float(per_class_f1[idx])

        return None

    except Exception:
        return None


def build_gradcam_prompt(top_disease, model_analyses, per_class_f1_info):
    return f"""
You are MedGemma analyzing GradCAM attention maps from multiple retinal disease models.

Top predicted disease:
{top_disease}

Model analyses:
{json.dumps(model_analyses, indent=2)}

Per-model F1 scores:
{json.dumps(per_class_f1_info, indent=2)}

Return valid JSON with:

{{
  "consensus_analysis": {{
    "agreement_level": "high | moderate | low | conflicting",
    "common_attention_regions": [],
    "clinical_plausibility": "high | moderate | low | uncertain",
    "consensus_explanation": "..."
  }},
  "model_insights": [
    {{
      "model": "...",
      "attention_quality": "...",
      "what_model_may_be_seeing": "...",
      "clinical_correlation": "..."
    }}
  ],
  "clinical_findings": {{
    "anatomical_structures": [],
    "pathological_features": [],
    "pattern_match_with_disease": "consistent | partial | inconsistent | uncertain",
    "explanation": "..."
  }},
  "confidence_assessment": {{
    "overall_confidence": "high | moderate | low | uncertain",
    "confidence_factors": [],
    "limitations": []
  }},
  "summary": "..."
}}

Important:
- Do not diagnose.
- Do not prescribe treatment.
- Output valid JSON only.
"""


def create_gradcam_interpretation_tool(medgemma_model):
    @tool(
        name_or_callable="gradcam_interpretation",
        args_schema=GradCamInterpretationInput,
    )
    def gradcam_interpretation(
        gradcam_images: dict[str, Any],
        top_disease: str,
        per_class_f1: list[float] | None = None,
        class_names: list[str] | None = None,
        top_disease_index: int | None = None,
    ) -> str:
        """
        Analyze GradCAM heatmaps from multiple retinal models.
        Uses MedGemma to interpret attention regions and clinical plausibility.
        Does not diagnose or prescribe treatment.
        """

        if not gradcam_images:
            return json.dumps(
                {
                    "tool_name": "gradcam_interpretation",
                    "error": "No GradCAM images provided.",
                },
                indent=2,
            )

        model_analyses = {}
        per_class_f1_info = {}

        for model_name, image_input in gradcam_images.items():
            try:
                gradcam_img = load_gradcam_image(image_input)

                f1_score = None

                if per_class_f1 is not None and len(per_class_f1) > 0:
                    if top_disease_index is not None:
                        if top_disease_index < len(per_class_f1):
                            f1_score = float(per_class_f1[top_disease_index])

                    elif class_names is not None and top_disease:
                        f1_score = get_model_f1_for_disease(
                            per_class_f1=per_class_f1,
                            class_names=class_names,
                            disease_name=top_disease,
                        )

                analysis = analyze_attention_regions(
                    gradcam_image=gradcam_img,
                    model_name=model_name,
                    top_disease=top_disease,
                    f1_score=f1_score,
                )

                model_analyses[model_name] = analysis
                per_class_f1_info[model_name] = analysis.get(
                    "model_performance_f1", "N/A"
                )

            except Exception as e:
                model_analyses[model_name] = {
                    "error": str(e),
                    "attention_percentage": "N/A",
                    "attention_location": "N/A",
                    "attention_focus": "N/A",
                    "attention_peaks": [],
                }
                per_class_f1_info[model_name] = "N/A"

        prompt = build_gradcam_prompt(
            top_disease=top_disease,
            model_analyses=model_analyses,
            per_class_f1_info=per_class_f1_info,
        )

        # `medgemma_model` is a callable: (prompt: str) -> str
        raw_output = medgemma_model(prompt)

        parsed_output = safe_parse_json(raw_output)

        result = {
            "tool_name": "gradcam_interpretation",
            "top_disease": top_disease,
            "model_analyses": model_analyses,
            "interpretation": parsed_output,
        }

        return json.dumps(result, indent=2)

    return gradcam_interpretation
