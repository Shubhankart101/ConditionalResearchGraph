from langchain_core.messages import AIMessage, ToolMessage

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": "Search local evidence for a research query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_EVIDENCE = {
    "langgraph": "LangGraph models agent workflows as stateful graphs of nodes and edges.",
    "reducer": "A reducer such as add_messages merges node updates into existing state.",
}


def search_evidence(query: str) -> str:
    hits = [text for key, text in _EVIDENCE.items() if key in query.lower()]
    return "\n".join(hits) if hits else "No evidence found."


def execute_tool_calls(message: AIMessage) -> list[ToolMessage]:
    results = []
    for call in message.tool_calls:
        if call["name"] == "search_evidence":
            content = search_evidence(call["args"].get("query", ""))
        else:
            content = f"Unknown tool: {call['name']}"
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
    return results
