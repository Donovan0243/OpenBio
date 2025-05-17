import os
import sys
from typing import List, Dict, Union, Generator, Iterator, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# 添加项目根目录到 Python 路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.core.settings import configure_settings
from src.core.rag import initialize_rag_system

class Pipeline:
    def __init__(self):
        self.name = "OpenBio RAG Pipeline"
        print("[OpenBio] 初始化 Pipeline...")
        self.workflow = None
        try:
            self.original_dir = os.getcwd()
            os.chdir(ROOT_DIR)
            print(f"[DEBUG] 切换到根目录: {ROOT_DIR} 进行初始化")
            configure_settings()
            self.workflow = initialize_rag_system()
            print("[OpenBio] Pipeline 初始化成功")
        except Exception as e:
            print(f"[OpenBio] 初始化错误: {str(e)}")
            raise
        finally:
            # 初始化后即恢复原始目录，除非有其他需要在 ROOT_DIR 执行的操作
            if hasattr(self, 'original_dir'):
                 os.chdir(self.original_dir)
                 print(f"[DEBUG] 初始化后恢复目录到: {self.original_dir}")


    async def on_startup(self):
        print("[OpenBio] Pipeline 启动")
        if self.workflow is None:
            print("[OpenBio] 警告: Workflow 未在 __init__ 中成功初始化!")
            # 此处可能需要更健壮的错误处理或状态检查

    async def on_shutdown(self):
        print("[OpenBio] Pipeline 关闭")


    def pipe(
        self, user_message: str, model_id: str, messages: List[Dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        original_cwd = os.getcwd()
        if ROOT_DIR != original_cwd:
            os.chdir(ROOT_DIR)
            # print(f"[DEBUG] pipe: 切换到根目录: {ROOT_DIR} 执行 workflow")

        if self.workflow is None:
            def error_gen():
                if ROOT_DIR != original_cwd: os.chdir(original_cwd)
                yield "[OpenBio] 错误: RAG 工作流未初始化。"
            return error_gen()

        try:
            inputs: Dict[str, Any] = {
                "messages": [
                    HumanMessage(content=user_message, additional_kwargs={"type": "user_question"})
                ]
            }

            def stream_workflow_responses():
                final_answer_content = None
                thinking_steps_started = False
                
                # type_labels 现在主要用于给来自 metadata.thinking_content 的文本配一个 emoji/前缀
                type_labels = {
                    "routing_info": "🧭 Routing", 
                    "evaluation_info": "⚖️ Evaluation",
                    "agent_step_info": "🐾 Agent Step", # 用于 eutils_agent, blast_agent 等的 thinking_content
                    "error_info": "❗ Error Info",
                    "general_thought": "🤔 Thinking", # 通用或未知节点的 thinking_content
                }
                
                print("[STREAM_DEBUG] Starting stream_workflow_responses generator (ONLY metadata.thinking_content for <think> block)...")

                for chunk_idx, chunk in enumerate(self.workflow.stream(inputs, {"recursion_limit": 25})):
                    print(f"\n[STREAM_DEBUG] Chunk {chunk_idx + 1}: {chunk}")
                    
                    for node_name, node_output_value in chunk.items():
                        print(f"[STREAM_DEBUG]   Processing Node: '{node_name}'")
                        if node_name == "__end__":
                            print("[STREAM_DEBUG]     Node is __end__, skipping.")
                            continue

                        # 1. 检查 'metadata' 中是否有 'thinking_content' 来生成思考步骤
                        if isinstance(node_output_value, dict) and "metadata" in node_output_value:
                            metadata_dict = node_output_value.get("metadata")
                            if isinstance(metadata_dict, dict) and "thinking_content" in metadata_dict:
                                custom_think_text = metadata_dict.get("thinking_content")
                                if custom_think_text and isinstance(custom_think_text, str) and custom_think_text.strip():
                                    print(f"[STREAM_DEBUG]     Found 'thinking_content' in metadata from '{node_name}': {custom_think_text}")
                                    
                                    if not thinking_steps_started:
                                        print("[STREAM_DEBUG]       Starting <think> block.")
                                        yield "<think>\n"
                                        thinking_steps_started = True
                                    
                                    label_key = "general_thought" # 默认标签
                                    if node_name == "router": label_key = "routing_info"
                                    elif node_name == "evaluator": label_key = "evaluation_info"
                                    elif "agent" in node_name.lower(): label_key = "agent_step_info"
                                    # 检查顶层是否有 status: error，来决定是否用 error_info 标签
                                    if isinstance(node_output_value, dict) and node_output_value.get("status") == "error":
                                        label_key = "error_info"
                                    
                                    label_prefix = type_labels.get(label_key, type_labels["general_thought"]) # 获取标签前缀
                                    line_to_yield = f"{label_prefix}: {custom_think_text.strip()}\n"
                                    print(f"[STREAM_DEBUG]         Yielding from thinking_content: {line_to_yield.strip()}")
                                    yield line_to_yield
                            else:
                                print(f"[STREAM_DEBUG]     'metadata' dict found for '{node_name}', but no 'thinking_content' key or it's empty.")
                        else:
                            print(f"[STREAM_DEBUG]     No 'metadata' key in output of node '{node_name}', or output is not a dict. Cannot check for thinking_content.")

                        # 2. 处理 'messages' 列表，但仅为了提取 'final_answer'
                        # 其他类型的消息将不再用于生成 <think> 块的内容
                        if isinstance(node_output_value, dict) and "messages" in node_output_value:
                            messages_in_node_output = node_output_value.get("messages", [])
                            if isinstance(messages_in_node_output, list):
                                print(f"[STREAM_DEBUG]     Checking {len(messages_in_node_output)} message(s) in '{node_name}' for final_answer.")
                                for msg_obj in messages_in_node_output:
                                    if isinstance(msg_obj, AIMessage) and hasattr(msg_obj, 'additional_kwargs') and msg_obj.additional_kwargs:
                                        msg_type = msg_obj.additional_kwargs.get("type")
                                        if msg_type == "final_answer":
                                            final_answer_content = str(msg_obj.content).strip()
                                            print(f"[STREAM_DEBUG]         Found final_answer in messages: {final_answer_content[:70]}...")
                                            # 不再从此循环 yield 非 final_answer 的内容到 <think> 块
                        elif isinstance(node_output_value, AIMessage): # 处理节点直接返回单个 AIMessage 的情况
                            msg_obj = node_output_value
                            if hasattr(msg_obj, 'additional_kwargs') and msg_obj.additional_kwargs:
                                msg_type = msg_obj.additional_kwargs.get("type")
                                if msg_type == "final_answer":
                                    final_answer_content = str(msg_obj.content).strip()
                                    print(f"[STREAM_DEBUG]         Found final_answer directly from node '{node_name}': {final_answer_content[:70]}...")


                if thinking_steps_started:
                    print("[STREAM_DEBUG] Ending <think> block.")
                    yield "</think>\n"
                
                if final_answer_content:
                    print(f"[STREAM_DEBUG] Yielding final answer: {final_answer_content[:50]}...")
                    yield f"Answer: {final_answer_content}"
                else:
                    print("[STREAM_DEBUG] No final_answer_content found after all chunks.")
                    yield "[OpenBio] 警告: 执行完成，但未找到明确的 'final_answer'。"
                
                if ROOT_DIR != original_cwd:
                    os.chdir(original_cwd)
                print("[STREAM_DEBUG] stream_workflow_responses generator finished.")

            return stream_workflow_responses()

        except Exception as e:
            import traceback
            error_str = str(e)
            print(f"[OpenBio Pipeline] 执行过程中捕获到异常: {error_str}")
            print(traceback.format_exc())
            if ROOT_DIR != original_cwd: os.chdir(original_cwd)
            def error_gen_exception():
                yield f"[OpenBio] 系统内部错误 (流式处理): {error_str}"
            return error_gen_exception()