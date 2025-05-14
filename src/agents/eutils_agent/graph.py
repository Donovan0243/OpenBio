from langgraph.graph import StateGraph, END
from .component import EutilsComponent
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义状态类型 - 与BLAST保持一致
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: dict  # 用于在节点间传递数据

def create_eutils_subgraph():
    """创建E-utilities查询流程的子图"""
    # 实例化E-utils组件
    eutils_component = EutilsComponent()
    
    # 创建子图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("init_search", eutils_component.init_search)
    workflow.add_node("fetch_details", eutils_component.fetch_details)
    
    # 设置入口点
    workflow.set_entry_point("init_search")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "init_search",
        lambda x: x.get("next", END) if "next" in x else END,
        {
            "fetch_details": "fetch_details",
            END: END
        }
    )
    
    # fetch_details完成后结束
    workflow.add_edge("fetch_details", END)
    
    # 编译子图
    return workflow.compile()