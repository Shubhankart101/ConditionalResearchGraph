from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from main import build_graph, make_agent_node, route_after_agent
from utils.local_tools import tools_node


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append(list(messages))
        return next(self.replies)


def tool_request(name="search_evidence", args=None):
    call = {"name": name, "args": args or {"query": "battery"}, "id": "c1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def run(replies, counter=None):
    llm = FakeLLM(replies)
    node = tools_node
    if counter is not None:
        def node(state):
            counter.append(1)
            return tools_node(state)

    app = build_graph(make_agent_node(llm), node)
    return app.invoke({"messages": [HumanMessage(content="q")]}), llm


def test_route_after_agent():
    assert route_after_agent({"messages": []}) == "end"
    assert route_after_agent({"messages": [AIMessage(content="hi")]}) == "end"
    assert route_after_agent({"messages": [HumanMessage(content="hi")]}) == "end"
    assert route_after_agent({"messages": [tool_request()]}) == "tools"


def test_direct_answer_skips_tools():
    calls = []
    result, _ = run([AIMessage(content="done")], calls)
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]
    assert not calls


def test_tool_loop_order_and_history():
    result, llm = run([tool_request(), AIMessage(content="final")])
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert len(llm.seen[1]) == 3
    assert "guide-battery" in result["messages"][2].content


def test_unknown_tool_returns_to_agent():
    result, _ = run([tool_request("nope"), AIMessage(content="sorry")])
    assert "unknown_tool" in result["messages"][2].content
    assert result["messages"][-1].content == "sorry"
