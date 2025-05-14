from langgraph.graph import StateGraph, END
from .component import BlastComponent
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: dict  # 添加metadata字段用于在节点间传递数据

def create_blast_subgraph():
    """创建具有完整BLAST查询流程的子图"""
    # 实例化BLAST组件
    blast_component = BlastComponent()
    
    # 创建子图
    workflow = StateGraph(AgentState)
    
    # 添加节点，移除analyze_results节点
    workflow.add_node("init_query", blast_component.init_blast_query)
    workflow.add_node("fetch_results", blast_component.fetch_blast_results)
    
    # 设置入口点
    workflow.set_entry_point("init_query")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "init_query",
        lambda x: x.get("next", END) if "next" in x else END,
        {
            "fetch_results": "fetch_results",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "fetch_results",
        lambda x: x.get("next", END) if "next" in x else END,
        {
            "fetch_results": "fetch_results",  # 循环获取结果，直到完成或超时
            END: END
        }
    )
    
    # 编译子图
    return workflow.compile()