# src/core/rag.py
from typing import Annotated, Sequence, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from .router import Router
from .evaluator import Evaluator
from .generator import Generator
from ..agents.eutils_agent.graph import create_eutils_subgraph
from ..agents.blast_agent.graph import create_blast_subgraph
from ..agents.search_agent.graph import create_search_subgraph

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    status: str 
    error: Optional[str]  # 添加错误信息字段
    metadata: Dict[str, Any]  # 添加元数据字段

def initialize_rag_system():
    # 创建组件
    router = Router()
    evaluator = Evaluator()
    generator = Generator()
    
    # 创建工作流
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("router", router.route)
    workflow.add_node("evaluator", evaluator.evaluate)
    workflow.add_node("generator", generator.generate)
    workflow.add_node("eutils_agent", create_eutils_subgraph())
    workflow.add_node("blast_agent", create_blast_subgraph())
    workflow.add_node("search_agent", create_search_subgraph())  # 添加搜索代理
    
    # 添加边
    workflow.add_edge(START, "router")  # 从START到router
    
    # Router 的条件路由
    workflow.add_conditional_edges(
        "router",
        lambda x: END if x.get("status") == "error" else x["next"],
        {
            "eutils_agent": "eutils_agent",
            "blast_agent": "blast_agent",
            "search_agent": "search_agent",  # 添加搜索代理的边
        }
    )
    
    # Agent 的条件路由
    for agent in ["eutils_agent", "blast_agent", "search_agent"]:  # 添加search_agent
        workflow.add_conditional_edges(
            agent,
            lambda x: END if x.get("status") == "error" else "evaluator",
            {
                "evaluator": "evaluator",
            }
        )
    
    # Evaluator 的条件路由
    workflow.add_conditional_edges(
        "evaluator",
        lambda x: (
            END if x.get("status") == "error" 
            else x["next"]  # "generate" 或 "router"
        ),
        {
            "generate": "generator",
            "router": "router",
        }
    )
    
    # Generator生成最终答案
    workflow.add_edge("generator", END)
    
    # 编译工作流
    graph = workflow.compile()
    
    # 打印Mermaid格式的图
    print("\nMermaid Graph:")
    print("```mermaid")
    print(graph.get_graph().draw_mermaid())
    print("```")
    
    return graph