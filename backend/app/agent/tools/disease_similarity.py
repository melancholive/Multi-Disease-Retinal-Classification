import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.json_parsing import safe_parse_json


class DiseaseSimilarityInput(BaseModel):
    top_diseases: list[dict[str, Any]] = Field(
        ..., description="Candidate eye diseases"
    )


def build_disease_similarity_prompt(top_diseases):
    return f"""
Return valid JSON only.

Candidate diseases:
{json.dumps(top_diseases, indent=2)}

Return this JSON schema:

{{
  "most_similar_pairs": [
    {{
      "diseases": ["...", "..."],
      "similarity_reason": "...",
      "how_they_may_be_confused": "...",
      "features_that_help_distinguish_them": "..."
    }}
  ],
  "overall_summary": "...",
  "clinical_caution": "Do not diagnose."
}}
"""


def create_disease_similarity_tool(medgemma_model):
    @tool(
        name_or_callable="disease_similarity_analysis",
        args_schema=DiseaseSimilarityInput,
    )
    def disease_similarity_analysis(top_diseases: list[dict[str, Any]]) -> str:
        """
        Use MedGemma to compare multiple candidate eye diseases.
        Explains similarities, shared mechanisms, co-occurrence, and likely diagnostic confusion.
        Does not diagnose.
        """

        if len(top_diseases) < 2:
            return json.dumps(
                {
                    "tool_name": "disease_similarity_analysis",
                    "top_diseases": top_diseases,
                    "error": "At least two candidate diseases are needed for similarity analysis.",
                },
                indent=2,
            )

        prompt = build_disease_similarity_prompt(top_diseases)

        raw_output = medgemma_model(prompt)
        parsed_output = safe_parse_json(raw_output)

        result = {
            "tool_name": "disease_similarity_analysis",
            "top_diseases": top_diseases,
            "similarity_analysis": parsed_output,
        }

        return json.dumps(result, indent=2)

    return disease_similarity_analysis
