import re
import logging
import time
import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from ...tools.call_api import call_api
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlastComponent:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:14b",
            base_url="http://34.142.153.30:11434",
            api_key="ollama",
            temperature=0
        )
    
    def is_duplicate_params(self, new_params: Dict[str, Any], used_params: List[Dict[str, Any]]) -> bool:
        """
        检查BLAST参数是否重复
        只检查sequence是否重复，hitlist_size可以不同
        """
        for old_params in used_params:
            # 只检查sequence是否相同
            if new_params.get('sequence') == old_params.get('sequence'):
                return True
        return False

    def init_blast_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第一步：初始化BLAST查询，生成PUT请求"""
        # 确保保留metadata
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取历史参数记录
        used_params = metadata.get("used_blast_params", [])
        
        # 获取所有用户消息
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("type") == "user_question"]
        
        # 获取之前的BLAST操作历史
        agent_history = [
            msg for msg in messages 
            if isinstance(msg, AIMessage) and 
            msg.additional_kwargs.get("type") in ["blast_progress", "blast_response", "eutils_progress", "eutils_response"]
        ]
        
        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": metadata
            }
        
        # 获取第一条用户问题
        original_question = user_question[0].content
        logger.info(f"初始化BLAST查询: {original_question}")
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 格式化已使用过的参数
        used_params_text = ""
        if used_params:
            used_params_text = "\nPreviously used sequences(for reference):\n"
            for i, params in enumerate(used_params, 1):
                used_params_text += f"{i}. {params['sequence'][:50]}...\n"
        
        # 构建单一提示
        combined_prompt = f"""

You are a parameter extractor for NCBI BLAST API. You need to extract the DNA sequence from the user's question.
If there were previous BLAST operations, review them to understand if a new query is needed or if we should work with existing results.

Here is an example:
Question: Align the DNA sequence to the human genome: ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT?
Output: {{"sequence": "ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT","hitlist_size": 10}}

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------
PREVIOUS USED SEQUENCES (for reference):
{used_params_text if used_params_text else "No previous used sequences."}
--------------------------------

Extract relevant search terms from the user's question and the previous history.
IMPORTANT: 
1. Only return a JSON object with these keys DIRECTLY, do NOT include any other text or comments.
2. Try to use a different sequence if the desired sequence has already been used, but accuracy is more important than uniqueness.
"""
        
        # 使用单一SystemMessage
        blast_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(blast_prompt)
            
            # 解析JSON响应
            try:
                params = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    logger.error(f"无法从LLM响应中提取有效的JSON: {response}")
                    return {
                        "messages": messages + [
                            AIMessage(content="Cannot extract valid JSON from LLM response",
                                    additional_kwargs={"type": "blast_error"})
                        ],
                        "status": "error",
                        "metadata": metadata
                    }
                params = json.loads(json_match.group(1))

            # 使用重复检查方法
            if self.is_duplicate_params(params, used_params):
                logger.warning(f"检测到重复的序列: {params['sequence'][:50]}...")
                return {
                    "messages": messages + [
                        AIMessage(content="This sequence has been used before. Please try with a different sequence or rephrase your question to use a new sequence.",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }

            # 验证必要的参数是否存在
            if "sequence" not in params:
                logger.error(f"缺少必要的序列参数: {params}")
                return {
                    "messages": messages + [
                        AIMessage(content="Missing required sequence parameter",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }

            # 构建BLAST URL
            url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML"
            url += f"&QUERY={params['sequence']}"
            if "hitlist_size" in params:
                url += f"&HITLIST_SIZE={params['hitlist_size']}"
            else:
                url += "&HITLIST_SIZE=10"  # 默认值

            logger.info(f"生成的BLAST URL: {url}")
            
            # 发起PUT请求
            api_response = call_api(url)
            if api_response is None:
                return {
                    "messages": messages + [
                        AIMessage(content="BLAST Put request failed",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
            
            # 提取RID
            rid_match = re.search('RID = (.*)\n', api_response.decode('utf-8'))
            if not rid_match:
                logger.error("无法从BLAST响应中提取RID")
                return {
                    "messages": messages + [
                        AIMessage(content="Could not extract RID from BLAST response",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
                
            rid = rid_match.group(1)
            
            # 记录使用过的参数
            used_params.append(params)
            
            # 返回结果，更新metadata
            return {
                "messages": messages + [
                    AIMessage(
                        content=f"Initiated BLAST query with: [{url}]\nReceived RID: {rid}",
                        additional_kwargs={
                            "type": "blast_progress",
                            "parameters": params  # 在消息中也保存参数，方便查看
                        }
                    )
                ],
                "next": "fetch_results",
                "metadata": {
                    **metadata,
                    "blast_rid": rid,
                    "attempt": 0,
                    "used_blast_params": used_params  # 更新使用过的参数列表
                }
            }
            
        except Exception as e:
            logger.error(f"BLAST查询初始化过程中出错: {str(e)}")
            return {
                "messages": messages + [
                    AIMessage(content=f"Error initializing BLAST query: {str(e)}",
                            additional_kwargs={"type": "blast_error"})
                ],
                "status": "error",
                "metadata": metadata
            }
    
    def fetch_blast_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第二步：获取BLAST查询结果"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取保存的RID
        rid = metadata.get("blast_rid")
        if not rid:
            logger.error("没有找到BLAST查询的RID")
            return {
                "messages": messages + [
                    AIMessage(content="No RID found for BLAST query",
                            additional_kwargs={"type": "blast_error"})
                ],
                "status": "error",
                "metadata": metadata
            }
        
        # 获取当前尝试次数
        attempt = metadata.get("attempt", 0)
        metadata["attempt"] = attempt + 1
        
        logger.info(f"尝试获取BLAST结果, RID: {rid}, 尝试次数: {attempt+1}")
        
        # 构建GET请求URL
        get_url = f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}"
        
        # 等待一段时间后再获取结果
        waiting_time = min(15 * (attempt + 1), 60)  # 随着尝试次数增加等待时间，最长60秒
        logger.info(f"等待 {waiting_time} 秒后获取结果...")
        time.sleep(waiting_time)
        
        # 发起GET请求
        api_response = call_api(get_url)
        
        if api_response is None:
            if attempt < 3:  # 最多尝试3次
                return {
                    "messages": messages + [
                        AIMessage(content=f"Waiting for BLAST results (attempt {attempt+1}/3)...",
                                additional_kwargs={"type": "blast_progress"})
                    ],
                    "next": "fetch_results",  # 再次尝试获取结果
                    "metadata": metadata
                }
            else:
                return {
                    "messages": messages + [
                        AIMessage(content="Failed to retrieve BLAST results after multiple attempts",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
        
        # 检查是否仍在运行
        response_text = api_response.decode('utf-8')
        if "Status=WAITING" in response_text or "is still running" in response_text:
            if attempt < 3:  # 最多尝试3次
                return {
                    "messages": messages + [
                        AIMessage(content=f"BLAST analysis still running (attempt {attempt+1}/3)...",
                                additional_kwargs={"type": "blast_progress"})
                    ],
                    "next": "fetch_results",  # 再次尝试获取结果
                    "metadata": metadata
                }
            else:
                return {
                    "messages": messages + [
                        AIMessage(content="BLAST analysis is taking too long, please try again later",
                                additional_kwargs={"type": "blast_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
        
        # 裁剪过长的结果
        if len(response_text) > 10000:
            response_text = response_text[:10000] + "... [result is truncated]"
        
        # 返回最终结果，但不进行分析
        return {
            "messages": messages + [
                AIMessage(
                    content=f"BLAST Results:\n\n{response_text}",
                    additional_kwargs={
                        "type": "blast_response",
                        "url": get_url,
                        "rid": rid
                    }
                )
            ],
            # 不再指向analyze_results
            "metadata": metadata
        }