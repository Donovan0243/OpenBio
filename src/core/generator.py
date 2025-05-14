from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:32b",
            base_url="http://34.142.153.30:11434",
            api_key="ollama",
            num_ctx=16000,
            temperature=0
        )
    
    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 保留metadata
        metadata = state.get("metadata", {})
        messages = state["messages"]

        # 获取所有用户消息
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

        agent_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        if not user_messages:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": metadata
            }
        
        # 获取第一条用户问题
        original_question = user_messages[0].content

        # 如果有eval_count，记录日志
        if metadata.get("eval_count"):
            logger.info(f"生成最终答案 (经过 {metadata['eval_count']} 轮评估)")
        else:
            logger.info("生成最终答案...")

        # 构建历史记录文本
        history_text = ""
        for msg in agent_messages:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 生成答案的提示词 - 单一提示形式
        combined_prompt = f"""
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS RESULTS AND ANALYSIS:
{history_text if history_text else "No previous analysis available."}
--------------------------------
You are a bioinformatician. Based on the above conversation history and research results, answer the user's question.
Please adhere to the following requirements:
1. The answer should be direct and clear
2. Don't add unnecessary foreshadowing
3. Use the information gathered from the tools (E-utils, BLAST, etc.) to provide an accurate response
4. If the information is insufficient, acknowledge the limitations

Please generate your final answer now.
"""
        
        # 调用LLM生成答案 - 使用单一提示
        response = self.llm.invoke([SystemMessage(content=combined_prompt)])
        
        logger.info("答案生成完成")
        
        # 返回生成的答案
        return {
            "messages": messages + [
                AIMessage(content=response.content)
            ],
            "metadata": metadata
        }
