import re
import json
import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from ...tools.call_api import call_api
# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class EutilsComponent:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:14b",
            base_url="http://34.142.153.30:11434",
            temperature=0,
            num_ctx=16000
        )
    
    def is_duplicate_params(self, new_params: Dict[str, Any], used_params: List[Dict[str, Any]]) -> bool:
        """
        检查参数是否重复
        规则：
        1. esearch阶段：检查db和term是否与之前任何一轮都相同
        2. efetch阶段：检查db和id是否与之前任何一轮都相同
        3. 两个阶段互不干扰
        """
        # 判断当前是哪个阶段
        is_esearch = 'term' in new_params
        is_efetch = 'id' in new_params
        
        for old_params in used_params:
            # 检查数据库是否相同
            if new_params.get('db') != old_params.get('db'):
                continue
            
            # esearch阶段：检查term
            if is_esearch and 'term' in old_params:
                if new_params['term'] == old_params['term']:
                    return True
                
            # efetch阶段：检查id
            if is_efetch and 'id' in old_params:
                new_ids = set(new_params['id'].split(','))
                old_ids = set(old_params['id'].split(','))
                if new_ids == old_ids:
                    return True
        
        return False

    def init_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第一步：初始化搜索，使用esearch API"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取历史参数记录
        used_params = metadata.get("used_eutils_params", [])
        
        # 获取用户问题
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("type") == "user_question"]

        # 获取系统消息  
        system_message = [msg for msg in messages if isinstance(msg, SystemMessage) and msg.additional_kwargs.get("type") == "system_prompt"]
        
        # 获取之前的E-utils操作历史
        agent_history = [
            msg for msg in messages 
            if isinstance(msg, AIMessage) and 
            msg.additional_kwargs.get("type") in ["eutils_progress", "eutils_response","blast_progress","blast_response"]
        ]
        
        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": {
                    **metadata,
                    "thinking_content": "No user question found"
                }
            }
        
        # 获取第一条用户问题
        original_question = user_question[0].content
        logger.info(f"初始化E-utilities搜索: {original_question}")
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 格式化已使用过的参数为JSON格式
        used_params_text = ""
        if used_params:
            used_params_text = "\nPreviously used parameters:\n"
            for i, params in enumerate(used_params, 1):
                used_params_text += f"{json.dumps(params, indent=2)}\n"
        
        # 构建单一提示
        combined_prompt = f"""
You are a parameter generator for NCBI E-utilities API. Generate the parameters needed for an initial esearch request to the NCBI E-utilities API.

DATABASE SELECTION RULES:
1. For gene names, symbols, aliases, or questions about gene function: use db=gene
2. For SNP (rs) IDs or questions about genetic variants: use db=snp
3. For genetic disorders, diseases, or phenotypes: use db=omim

Here are some examples:
Question: What is the official gene symbol of LMP10?
Output: {{"db": "gene", "term": "LMP10", "retmax": 5}}

Question: Which gene is SNP rs1217074595 associated with?
Output: {{"db": "snp", "term": "rs1217074595", "retmax": 10}}

Question: What are genes related to Meesmann corneal dystrophy?
Output: {{"db": "omim", "term": "Meesmann corneal dystrophy", "retmax": 20}}

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------
PREVIOUS USED PARAMETERS:
{used_params_text if used_params_text else "No previous used parameters."}
--------------------------------

Extract relevant search terms from the user's question and the previous history.
IMPORTANT: 
1. First review the previous search history and parameters to understand what has been tried
2. Try to use different combinations of parameters if the desired parameter has already been used:
   - Use different search terms in the same database
   - Try different databases for the same search term
3. However, if the user's question clearly refers to specific parameters, prioritize accuracy over trying new combinations
4. Only return a JSON object with these keys DIRECTLY, do NOT include any other text or comments.
"""
        
        # 使用单一SystemMessage
        search_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(search_prompt)
            
            # 解析JSON响应
            try:
                params = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    raise ValueError("无法从LLM响应中提取有效的JSON")
                params = json.loads(json_match.group(1))
            
            # 使用通用参数检查方法
            if self.is_duplicate_params(params, used_params):
                logger.warning(f"检测到重复的参数: {params}")
                return {
                    "messages": messages + [
                        AIMessage(content="Duplicate parameters detected (same database and term), please try with different parameters or a different database",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": {
                        **metadata,
                        "thinking_content": "Duplicate parameters detected (same database and term)."
                    }
                }
            
            # 验证必要的参数是否存在
            if not all(key in params for key in ["db", "term"]):
                raise ValueError("缺少必要的参数: db 或 term")
            
            # 构建URL
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={params['db']}&term={params['term']}&retmode=json&sort=relevance"
            
            # 添加可选参数
            if "retmax" in params:
                url += f"&retmax={params['retmax']}"
            else:
                url += "&retmax=10"  # 默认值
            
            logger.info(f"生成的esearch URL: {url}")
            
            # 调用API
            api_response = call_api(url)
            if api_response is None:
                return {
                    "messages": messages + [
                        AIMessage(content=f"E-utilities esearch API call failed: {url}",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": {
                        **metadata,
                        "thinking_content": "E-utilities esearch API调用失败"
                    }
                }
            
            # 处理结果
            if isinstance(api_response, bytes):
                api_response = api_response.decode('utf-8')
            
            # 限制响应长度
            if len(api_response) > 10000:
                api_response = api_response[:10000] + "... [result is truncated]"
            
            # 记录使用过的参数
            used_params.append(params)
            
            # 返回结果，更新metadata
            return {
                "messages": messages + [
                    AIMessage(
                        content=f"[{url}]->\n[{api_response}]",
                        additional_kwargs={"type": "eutils_progress", "parameters": params}
                    )
                ],
                "next": "fetch_details",
                "metadata": {
                    **metadata,
                    "used_eutils_params": used_params,  # 更新使用过的参数列表
                    "thinking_content": f"E-utilities esearch results fetched:{api_response}.\n\n by calling {url}"
                }
            }
            
        except Exception as e:
            logger.error(f"E-utilities esearch过程中出错: {str(e)}")
            return {
                "status": "error",
                "error": f"E-utilities esearch error: {str(e)}",
                "metadata": {
                    **metadata,
                    "thinking_content": f"E-utilities esearch error: {str(e)}"
                }
            }
    
    def fetch_details(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第二步：获取详细信息，使用efetch或esummary API"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取历史参数记录
        used_params = metadata.get("used_eutils_params", [])
        
        # 所有历史消息
        eutils_responses = [
            msg for msg in messages
            if isinstance(msg, AIMessage) and 
            msg.additional_kwargs.get("type") in ["eutils_progress", "eutils_response"]
        ]
        
        # 获取用户问题
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("type") == "user_question"]
        if not user_question:
            return {
                "status": "error",
                "error": "No user question found",
                "metadata": {
                    **metadata,
                    "thinking_content": "No user question found"
                }
            }

        # 获取第一条用户问题
        original_question = user_question[0].content
        logger.info(f"获取E-utilities详细信息: {original_question}")
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in eutils_responses:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 格式化已使用过的参数为JSON格式
        used_params_text = ""
        if used_params:
            used_params_text = "\nPreviously used parameters:\n"
            for i, params in enumerate(used_params, 1):
                used_params_text += f"{json.dumps(params, indent=2)}\n"
        
        # 构建单一提示
        combined_prompt = f"""
You are a parameter generator for NCBI E-utilities API. Based on the previous esearch results shown above, generate the parameters needed for an efetch or esummary request.

Here are some examples:
Previous result contains IDs from gene database:
Output: {{"method": "efetch", "db": "gene", "id": "19171,5699,8138"}}

Previous result contains IDs from snp database:
Output: {{"method": "esummary", "db": "snp", "id": "1217074595"}}

Previous result contains IDs from omim database:
Output: {{"method": "esummary", "db": "omim", "id": "618767,601687,300778,148043,122100"}}

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------
PREVIOUS USED PARAMETERS:
{used_params_text}
--------------------------------

Extract the IDs from the previous esearch result and include them in your JSON.
IMPORTANT: 
1. First review the previous search history and parameters to understand what has been tried
2. Try to use different combinations of parameters if the desired parameter has already been used:
   - Extract different IDs from the same esearch result
3. However, if the user's question clearly refers to specific IDs, prioritize accuracy over trying new combinations
4. Only return a JSON object with these keys DIRECTLY, do NOT include any other text or comments.
"""
        
        # 使用单一SystemMessage
        fetch_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(fetch_prompt)
            
            # 解析JSON响应
            try:
                params = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    raise ValueError("无法从LLM响应中提取有效的JSON")
                params = json.loads(json_match.group(1))
            
            # 使用通用参数检查方法
            if self.is_duplicate_params(params, used_params):
                logger.warning(f"检测到重复的参数: {params}")
                return {
                    "messages": messages + [
                        AIMessage(content="Duplicate parameters detected (same database and IDs), please try with different parameters or a different database",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": {
                        **metadata,
                        "thinking_content": "Duplicate parameters detected (same database and IDs)."
                    }
                }
            
            # 验证必要的参数是否存在
            if not all(key in params for key in ["method", "db", "id"]):
                raise ValueError("缺少必要的参数: method, db 或 id")
            
            # 构建URL
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{params['method']}.fcgi?db={params['db']}&id={params['id']}&retmode=json&sort=relevance"
            
            # 添加可选参数
            if "retmax" in params:
                url += f"&retmax={params['retmax']}"
            
            logger.info(f"生成的{params['method']} URL: {url}")
            
            # 调用API
            api_response = call_api(url)
            if api_response is None:
                return {
                    "messages": messages + [
                        AIMessage(content=f"E-utilities API call failed: {url}",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": {
                        **metadata,
                        "thinking_content": "E-utilities API call failed"
                    }
                }
            
            # 处理结果
            if isinstance(api_response, bytes):
                api_response = api_response.decode('utf-8')
            
            # 限制响应长度
            if len(api_response) > 10000:
                api_response = api_response[:10000] + "... [result is truncated]"
            
            # 记录使用过的参数
            used_params.append(params)
            
            # 返回最终结果
            return {
                "messages": messages + [
                    AIMessage(
                        content=f"[{url}]->\n[{api_response}]",
                        additional_kwargs={"type": "eutils_response", "parameters": params}
                    )
                ],
                "metadata": {
                    **metadata,
                    "used_eutils_params": used_params,  # 需要更新metadata
                    "thinking_content": f"E-utilities details: {api_response}"
                },
            }
            
        except Exception as e:
            logger.error(f"E-utilities fetch过程中出错: {str(e)}")
            return {
                "status": "error",
                "error": f"E-utilities fetch error: {str(e)}",
                "metadata": metadata,
                "thinking_content": f"E-utilities fetch过程中出错: {str(e)}"
            }

    def format_eutils_history(self, eutils_history: List[AIMessage]) -> str:
        return "\n".join([f"[{msg.content}]" for msg in eutils_history])