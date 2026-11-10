"""Assemble a Conditional Research Graph.

Scenario
--------
A local research assistant can either answer directly or call a supplied
evidence-search tool before answering. The Groq client, model node, evidence
tool, tool-execution node, and test fixtures are already provided.

Your task
---------
Complete only the three TODO locations in this file:

1. Define the messages field in ResearchState.
2. Implement route_after_agent.
3. Implement build_graph.

When the application is run normally, LLMClient uses GROQ_API_KEY. During
evaluation, pytest injects a mocked LLMClient, so no real API call, API key, or
network access is used.

Required behavior
-----------------
ResearchState:
- Define messages as an append-only LangGraph message list.
- Use Annotated, AnyMessage, and the supplied add_messages reducer so node
  updates are added to the existing history instead of replacing it.

route_after_agent:
- Return "tools" when the latest AIMessage contains one or more tool calls.
- Return "end" when the latest message has no tool call.
- Return "end" for an empty message list.

build_graph:
- Create a StateGraph using ResearchState.
- Register the supplied callables under the node names "agent" and "tools".
- Connect START to "agent".
- Add a conditional edge after "agent" that maps "tools" to the tools node
  and "end" to END.
- Connect "tools" back to "agent".
- Compile and return the graph.

The evaluator checks message accumulation, both routing outcomes, a direct
answer, and a complete model-to-tool-to-model cycle. Do not modify the supplied
Groq client, model-node factory, or local tool implementation.

Not required
------------
You do not need to write prompts, implement the evidence tool, call a real
model during evaluation, configure LangSmith, build a user interface, or deploy
an application.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph, add_messages

from utils.llm_client import LLMClient
from utils.local_tools import SEARCH_TOOL_SCHEMA


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def make_agent_node(
    llm_client: LLMClient | None = None,
) -> Callable[[ResearchState], dict[str, list[AIMessage]]]:
    """Create the supplied model node for real or mocked LLM execution."""
    client = llm_client or LLMClient()

    def agent_node(state: ResearchState) -> dict[str, list[AIMessage]]:
        message = client.complete(
            list(state["messages"]),
            [SEARCH_TOOL_SCHEMA],
        )
        return {"messages": [message]}

    return agent_node


def route_after_agent(state: ResearchState) -> Literal["tools", "end"]:
    """Choose tool execution or graph completion from the latest message."""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    latest = messages[-1]
    if isinstance(latest, AIMessage) and latest.tool_calls:
        return "tools"
    return "end"


def build_graph(
    agent_node: Callable[[ResearchState], dict[str, Any]],
    tools_node: Callable[[ResearchState], dict[str, Any]],
) -> Any:
    """Build and compile the conditional research graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
