from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from main import build_graph, route_after_agent
from utils.local_tools import execute_tool_calls


def make_agent(script):
    replies = iter(script)

    def agent(state):
        return {"messages": [next(replies)]}

    return agent


def tools(state):
    return {"messages": execute_tool_calls(state["messages"][-1])}


def tool_call(name):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"query": "reducer"}, "id": "c1"}])


def test_route():
    assert route_after_agent({"messages": []}) == "end"
    assert route_after_agent({"messages": [AIMessage(content="hi")]}) == "end"
    assert route_after_agent({"messages": [tool_call("search_evidence")]}) == "tools"


def test_direct_answer_skips_tools():
    calls = []

    def spy_tools(state):
        calls.append(1)
        return tools(state)

    app = build_graph(make_agent([AIMessage(content="done")]), spy_tools)
    result = app.invoke({"messages": [HumanMessage(content="hi")]})
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]
    assert not calls


def test_tool_loop_order():
    app = build_graph(make_agent([tool_call("search_evidence"), AIMessage(content="final")]), tools)
    result = app.invoke({"messages": [HumanMessage(content="q")]})
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage, ToolMessage, AIMessage]


def test_unknown_tool_returns_to_agent():
    app = build_graph(make_agent([tool_call("nope"), AIMessage(content="sorry")]), tools)
    result = app.invoke({"messages": [HumanMessage(content="q")]})
    assert "Unknown tool" in result["messages"][2].content
    assert result["messages"][-1].content == "sorry"
