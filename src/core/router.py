from langchain_ollama import ChatOllama
import logging
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
logger = logging.getLogger(__name__)

class Router:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:32b",
            base_url="http://34.142.153.30:11434",
            api_key="ollama",
            num_ctx=16000,
            temperature=0
        )
    
    def route(self, state):
        # 保留metadata
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取用户问题
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage)]
        
        # 获取历史记录
        agent_history = [
            msg for msg in messages
            if isinstance(msg, AIMessage) and
            (msg.additional_kwargs.get("type") in ["eutils_response", "eutils_progress", "blast_response", "blast_progress", "search_response"])
        ]
        
        if not user_question:
            return {
                "status": "error",
                "error": "没有找到用户问题",
                "metadata": metadata
            }
        
        # 提取用户问题文本
        original_question = user_question[0].content
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 获取evaluator的意见
        eval_result = metadata.get("eval_result", {})
        eval_reason = eval_result.get("reason", "No evaluation reason provided")
        
        # 构建单一提示

        combined_prompt = f"""
You are a strict router that uses JSON to make routing decisions. You should analyze the conversation history above to make an informed decision about which agent to use.
router options:
   - eutils_agent: query the database to get the detail information about gene, protein, disease.
   - blast_agent: check the DNA sequence alignment and comparison.
   - search_agent: search the web when the question cannot be answered by eutils or blast.
   - irrelevant_questions: the question is not related to bioinformatics.

                                     
You should consider the following:
   - What is the question?
   - What information we have gathered so far?
   - Which tool would be most appropriate for the next step?
   - The evaluator's opinion.

--------------------------------
USER QUESTION: 
{original_question}
--------------------------------
CONVERSATION HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------
PREVIOUS EVALUATOR'S OPINION:
{eval_reason}
--------------------------------

You MUST output your decision in the following JSON format:
{{
    "agent": "eutils_agent" or "blast_agent" or "search_agent" or "irrelevant_questions",
    "reason": "Brief explanation of why this agent was chosen for the next step"
}}
                                     
Do not include any other text or formatting. ONLY return the JSON object.
"""
        
        # 构建单一消息的提示
        router_prompt = [SystemMessage(content=combined_prompt)]
        
        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm.invoke(router_prompt)
            
            try:
                # 尝试解析JSON响应
                response_json = json.loads(response.content)
                
                # 验证JSON格式是否正确
                if "agent" in response_json and "reason" in response_json:
                    agent = response_json["agent"]
                    reason = response_json["reason"]
                    
                    if agent in ["eutils_agent", "blast_agent", "search_agent","irrelevant_questions"]:
                        # 记录路由决策和原因
                        logger.info(f"路由决策: {agent}, 原因: {reason}")
                        return {
                            "next": agent,
                            "metadata": {
                                **metadata,
                                "routing_reason": "IRRELEVANT REQUEST." if agent == "irrelevant_questions" else reason
                            }
                        }
                    else:
                        logger.warning(f"未知的agent: {agent}")
                else:
                    logger.warning(f"JSON响应格式不正确: {response_json}")
            except json.JSONDecodeError:
                logger.warning(f"尝试 {attempt + 1}/{max_retries}: 无法解析JSON: {response.content}")
            
            logger.warning(f"尝试 {attempt + 1}/{max_retries}: {response.content}")
        
        # 默认使用 eutils
        logger.warning("多次尝试失败，默认使用 eutils")
        return {
            "next": "eutils_agent",
            "metadata": metadata
        }
    