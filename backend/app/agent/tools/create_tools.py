from app.agent.tools.disease_similarity import create_disease_similarity_tool
from app.agent.tools.gradcam_interpretation import create_gradcam_interpretation_tool
from app.agent.tools.next_step_recommendation import (
    create_next_step_recommendation_tool,
)
from app.services.medgemma import medgemma_model


def get_tools():
    next_step_recommendation_tool = create_next_step_recommendation_tool(medgemma_model)
    gradcam_interpretation_tool = create_gradcam_interpretation_tool(medgemma_model)
    disease_similarity_tool = create_disease_similarity_tool(medgemma_model)
    return [
        next_step_recommendation_tool,
        gradcam_interpretation_tool,
        disease_similarity_tool,
    ]
