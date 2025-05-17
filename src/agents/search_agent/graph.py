from langgraph.graph import StateGraph, END
from .component import SearchComponent
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义状态类型 - 与BLAST保持一致
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: dict  # 用于在节点间传递数据

def create_search_subgraph():
    """创建search_agent的子图"""
    search_component = SearchComponent()
    workflow = StateGraph(AgentState)
    workflow.add_node("init_search", search_component.init_search)
    workflow.set_entry_point("init_search")
    workflow.add_edge("init_search", END)
    return workflow.compile()