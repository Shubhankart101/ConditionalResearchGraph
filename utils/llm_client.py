import os

from langchain_groq import ChatGroq

from utils.local_tools import SEARCH_TOOL_SCHEMA


def get_model():
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    return llm.bind_tools([SEARCH_TOOL_SCHEMA])
