from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools.create_tools import get_tools

tools = get_tools()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """
You are a retinal imaging ReAct agent.

You may call tools when they are useful to answer the user's question.
The user may provide model prediction context including candidate diseases (top_k/top_diseases)
and optionally Grad-CAM overlays.

Important:
- image_path is the original retinal fundus image path.
- gradcam_images is separate from image_path.
- Do not treat image_path as gradcam_images.
- gradcam_images must always be a dictionary mapping model names to GradCAM image paths.
- Correct gradcam_images format:
{
  "efficientnet_b0": "/content/efficientnet_gradcam.png",
  "resnet50": "/content/resnet_gradcam.png",
  "shufflenet": "/content/shufflenet_gradcam.png"
}
- Never pass gradcam_images as a plain string.
- If the user provides only a single GradCAM image path, convert it into a dictionary using the model name as the key.
- Example:
{
  "gradcam_images": {
    "efficientnet_b0": "/content/efficientnet_gradcam.png"
  },
  "top_disease": "diabetic_retinopathy"
}

Safety rules:
- Do not diagnose.
- Do not prescribe treatment.
- Clearly state that model outputs and Grad-CAM findings require clinical review by an eye care professional.
"""


def model_node(state: AgentState):
    # Always prepend the system prompt so the agent has consistent safety/tool guidance.
    response = llm_with_tools.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)
graph = StateGraph(AgentState)

graph.add_node("agent", model_node)
graph.add_node("tools", tool_node)
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
agent_graph = graph.compile()
