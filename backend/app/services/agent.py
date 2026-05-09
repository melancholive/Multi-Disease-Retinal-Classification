from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agent.state import AgentState


class RetinalAgentService:
    def __init__(self, graph=None):
        self.graph = graph

    def get_graph(self):
        if self.graph is None:
            from app.agent.graph import get_agent_graph

            self.graph = get_agent_graph()
        return self.graph

    def normalize_gradcam_images(
        self,
        gradcam_images: dict[str, str] | str | None,
        default_model_name: str | None = None,
    ) -> dict[str, str] | None:
        if gradcam_images is None:
            return None

        if isinstance(gradcam_images, dict):
            return gradcam_images

        if isinstance(gradcam_images, str):
            key = default_model_name or "unknown_model"
            return {key: gradcam_images}

        raise ValueError("gradcam_images must be a dict, string, or None")

    def normalize_top_diseases(
        self,
        top_diseases: list[dict[str, Any]] | list[str] | None,
    ) -> list[dict[str, Any]] | None:
        if top_diseases is None:
            return None

        normalized = []
        for item in top_diseases:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"label": item})
            else:
                raise ValueError("top_diseases must contain strings or objects")

        return normalized

    def run(
        self,
        *,
        message: str,
        session_id: str,
        image_path: str | None = None,
        top_diseases: list[dict[str, Any]] | list[str] | None = None,
        top_k: int | None = None,
        prediction_context: dict | None = None,
        gradcam_images: dict[str, str] | str | None = None,
        default_gradcam_model: str | None = None,
    ):
        normalized_gradcams = self.normalize_gradcam_images(
            gradcam_images,
            default_model_name=default_gradcam_model,
        )
        normalized_top_diseases = self.normalize_top_diseases(top_diseases)

        input_state = cast(
            AgentState,
            {
                "messages": [HumanMessage(content=message)],
                "image_path": image_path,
                "top_diseases": normalized_top_diseases,
                "top_k": top_k,
                "prediction_context": prediction_context,
                "gradcam_images": normalized_gradcams,
            },
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        result = self.get_graph().invoke(input_state, config=config)

        final_message = result["messages"][-1]
        serialized_messages = []
        for message_item in result["messages"]:
            if hasattr(message_item, "model_dump"):
                serialized_messages.append(message_item.model_dump())
            elif isinstance(message_item, dict):
                serialized_messages.append(message_item)
            else:
                serialized_messages.append(
                    {
                        "type": getattr(message_item, "type", None),
                        "content": getattr(message_item, "content", str(message_item)),
                    }
                )

        return {
            "answer": final_message.content,
            "messages": serialized_messages,
        }
