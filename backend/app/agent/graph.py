import copy
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.state import AgentState
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools.create_tools import get_tools


def _summarize_gradcam_images(
    gradcam_images: dict[str, str] | None,
) -> dict[str, Any] | None:
    if not gradcam_images:
        return None

    return {
        "available": True,
        "models": sorted(
            model_name for model_name, image in gradcam_images.items() if image
        ),
    }


def _strip_large_image_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"cams", "overlay_png"}:
                continue
            cleaned[key] = _strip_large_image_payloads(item)
        return cleaned

    if isinstance(value, list):
        return [_strip_large_image_payloads(item) for item in value]

    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image data omitted]"

    return value


def _build_context(state: AgentState) -> dict[str, Any]:
    prediction_context = state.get("prediction_context")
    gradcam_images = state.get("gradcam_images")

    context = {
        key: value
        for key, value in {
            "image_path": state.get("image_path"),
            "top_diseases": state.get("top_diseases"),
            "top_k": state.get("top_k"),
            "prediction_context": (
                _strip_large_image_payloads(copy.deepcopy(prediction_context))
                if prediction_context is not None
                else None
            ),
            "gradcam_images": _summarize_gradcam_images(gradcam_images),
        }.items()
        if value is not None
    }

    return context


def build_agent_graph():
    tools = get_tools()

    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(tools)

    def model_node(state: AgentState):
        context = _build_context(state)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.extend(state["messages"])

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    graph = StateGraph(AgentState)

    graph.add_node("agent", model_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge("tools", "agent")

    memory = MemorySaver()

    return graph.compile(checkpointer=memory)


@lru_cache(maxsize=1)
def get_agent_graph():
    return build_agent_graph()
