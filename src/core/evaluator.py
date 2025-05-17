from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage,HumanMessage, ToolMessage
import logging
from typing import Dict, Any
from textwrap import dedent
import json
import re

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:32b",
            base_url="http://34.142.153.30:11434",
            api_key="ollama",
            num_ctx=16000,
            temperature=0
        )
        self.prompt_template2 = """\
You are a strict answer evaluator.
You should may a decision that whether the conversation history can answer the question.
- If the existing conversation history can answer the question, output GENERATE.
- If the existing conversation history cannot answer the question, output CONTINUE
- For the sequence gene alias question, we need to use blast_agent to get the gene symbol first and then use eutils_agent to get the gene alias.
"""

    
    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 获取或初始化评估计数器
        if "metadata" not in state:
            state["metadata"] = {}
        
        if "eval_count" not in state["metadata"]:
            state["metadata"]["eval_count"] = 0
        
        # 递增评估计数器
        state["metadata"]["eval_count"] += 1
        eval_count = state["metadata"]["eval_count"]
        
        # 检查是否达到评估轮数限制
        if eval_count >= 5:
            logger.info(f"已达到评估轮数上限 ({eval_count}/5)，强制进入生成阶段")
            return {
                "next": "generate",
                "metadata": state["metadata"]
            }

        messages = state["messages"]
        # 获取所有用户消息
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("type") == "user_question"]

        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": state["metadata"]
            }
        
        # 获取第一条用户问题
        original_question = user_question[0].content

        # 获取系统消息
        system_message = [msg for msg in messages if isinstance(msg, SystemMessage) and msg.additional_kwargs.get("type") == "system_prompt"]

        # 获取相关历史消息
        agent_history = [
            msg for msg in messages
            if (
                (isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") in [
                    "blast_progress", "blast_response",
                    "eutils_progress", "eutils_response",
                    "search_response"
                ])
                or isinstance(msg, ToolMessage)
            )
        ]

        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"

        
        # 使用LLM评估信息是否足够

        eval_prompt = [
        SystemMessage(content=f"""\
You are a strict answer evaluator.
You should make a decision about whether the conversation history can answer the question.
You need to return a JSON object with the following structure:
{{
    "next_step": "GENERATE" or "CONTINUE",
    "reason": "Brief explanation of your decision"
}}

Examples:
1. When information is sufficient:
{{
    "next_step": "GENERATE",
    "reason": "Found the official gene symbol Psmb10 in the results"
}}

2. When information is insufficient:
{{
    "next_step": "CONTINUE",
    "reason": "Only have gene IDs, need detailed gene information"
}}

--------------------------------
USER QUESTION: {original_question}
--------------------------------
CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation history."}
--------------------------------
Please return your decision as a JSON object.
""")
        ]
        
        try:
            response = self.llm.invoke(eval_prompt)
            
            # 解析JSON响应
            try:
                eval_result = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    logger.error(f"无法从评估响应中提取有效的JSON: {response}")
                    return {
                        "next": "router",
                        "metadata": {
                            **state["metadata"],
                            "eval_error": "Invalid JSON response from evaluator"
                        }
                    }
                eval_result = json.loads(json_match.group(1))
            
            logger.info(f"Evaluator决策: {eval_result} (评估轮数: {eval_count}/5)")
            
            if eval_result["next_step"] == "CONTINUE":
                logger.info(f"信息不足，返回router继续查询。原因: {eval_result['reason']}")
                return {
                    "next": "router",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": eval_result
                    }
                }
            elif eval_result["next_step"] == "GENERATE":
                logger.info(f"信息足够，进入生成阶段。原因: {eval_result['reason']}")
                return {
                    "next": "generate",
                    "metadata": {
                        **state["metadata"],
                        "eval_result": eval_result
                    }
                }
            else:
                logger.info("输出不规范，返回router继续查询")
                return {
                    "next": "router",
                    "metadata": {
                        **state["metadata"],
                        "eval_error": "Invalid decision from evaluator"
                    }
                }
            
        except Exception as e:
            logger.error(f"评估过程中出错: {str(e)}")
            return {
                "next": "router",
                "metadata": {
                    **state["metadata"],
                    "eval_error": f"Evaluation error: {str(e)}"
                }
            }
