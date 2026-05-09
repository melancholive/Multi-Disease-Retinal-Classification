from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    image_path: Optional[str]
    top_diseases: Optional[list[dict[str, Any]]]
    top_k: Optional[int]
    prediction_context: Optional[dict[str, Any]]

    gradcam_images: Optional[dict[str, str]]
