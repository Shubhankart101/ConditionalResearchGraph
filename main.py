from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def route_after_agent(state: ResearchState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "end"
    if getattr(messages[-1], "tool_calls", None):
        return "tools"
    return "end"


def build_graph(agent_node, tools_node):
    graph = StateGraph(ResearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def agent_node(state: ResearchState) -> dict:
    from utils.llm_client import get_model

    return {"messages": [get_model().invoke(state["messages"])]}


def tools_node(state: ResearchState) -> dict:
    from utils.local_tools import execute_tool_calls

    return {"messages": execute_tool_calls(state["messages"][-1])}


if __name__ == "__main__":
    app = build_graph(agent_node, tools_node)
    result = app.invoke({"messages": [("user", "What is a reducer in LangGraph?")]})
    for m in result["messages"]:
        print(type(m).__name__, ":", m.content)
