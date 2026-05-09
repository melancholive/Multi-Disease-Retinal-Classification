import json
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from app.utils.json_parsing import safe_parse_json


class NextStepRecommendationInput(BaseModel):
    top_diseases: list[dict[str, Any]] = Field(
        ..., description="Candidate eye diseases from the prediction"
    )


def build_next_step_prompt(
    top_diseases, similarity_analysis=None, patient_context=None
):
    return f"""
Return valid JSON only.

Candidate diseases:
{json.dumps(top_diseases, indent=2)}

Return this JSON schema:

{{
  "urgency_level": "routine | soon | urgent | emergency | uncertain",
  "recommended_next_steps": [
    {{
      "step": "...",
      "reason": "...",
      "priority": "low | medium | high | urgent"
    }}
  ],
  "additional_information_needed": ["..."],
  "patient_facing_explanation": "...",
  "clinical_caution": "Do not diagnose or prescribe treatment."
}}
"""


def create_next_step_recommendation_tool(medgemma_model):
    @tool(name_or_callable="next_step_recommendation")
    def next_step_recommendation(
        top_diseases: Annotated[
            list[dict[str, Any]] | None, InjectedState("top_diseases")
        ],
    ) -> str:
        """
        Use MedGemma to recommend safe next steps after candidate eye diseases are identified.
        This tool does not diagnose, prescribe treatment, or recommend medication changes.
        """

        if not top_diseases:
            return json.dumps(
                {
                    "tool_name": "next_step_recommendation",
                    "error": "No candidate diseases were provided in agent state.",
                },
                indent=2,
            )

        prompt = build_next_step_prompt(
            top_diseases=top_diseases,
        )

        raw_output = medgemma_model(prompt)

        parsed_output = safe_parse_json(raw_output)

        result = {
            "tool_name": "next_step_recommendation",
            "top_diseases": top_diseases,
            "next_step_recommendations": parsed_output,
        }

        return json.dumps(result, indent=2)

    return next_step_recommendation
