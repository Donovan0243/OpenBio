import re
import json
import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from ...tools.call_api import call_api
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EutilsComponent:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:32b",
            base_url="http://34.142.153.30:11434",
            temperature=0,
            num_ctx=16000
        )
    
    def init_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第一步：初始化搜索，使用esearch API"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
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
                "metadata": metadata
            }
        
        # 获取第一条用户问题
        original_question = user_question[0].content
        logger.info(f"初始化E-utilities搜索: {original_question}")
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 构建单一提示
        combined_prompt = f"""

You are a parameter generator for NCBI E-utilities API. Generate the parameters needed for an initial esearch request to the NCBI E-utilities API.

DATABASE SELECTION RULES:
1. For gene names, symbols, aliases, or questions about gene function: use db=gene
2. For DNA sequences: DNA sequences should NOT be directly searched in E-utils! They require BLAST first.
3. For SNP (rs) IDs or questions about genetic variants: use db=snp
4. For genetic disorders, diseases, or phenotypes: use db=omim


Here are some examples:
Question: What is the official gene symbol of LMP10?
Output: {{"db": "gene", "term": "LMP10", "retmax": 5}}

Question: Which gene is SNP rs1217074595 associated with?
Output: {{"db": "snp", "term": "rs1217074595", "retmax": 10}}

Question: What are genes related to Meesmann corneal dystrophy?
Output: {{"db": "omim", "term": "Meesmann corneal dystrophy", "retmax": 20}}


IMPORTANT GUIDELINES:
- If the question involves a DNA sequence, DO NOT use the sequence as a search term in E-utils
- For questions asking about aliases of genes containing DNA sequences, you should WAIT until BLAST results identify the gene
- Read the question carefully to determine what type of biological entity is being asked about

Only return a JSON object with these keys:
- db: The database to search (gene, snp, omim)
- term: The search term
- retmax: Number of results to return (recommend 5-20)

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------

Extract relevant search terms from the user's question and the previous history.
IMPORTANT: only return the JSON object, nothing else.
"""
        
        # 使用单一SystemMessage
        search_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(search_prompt)
            
            # 解析JSON响应
            try:
                # 尝试直接解析整个响应
                params = json.loads(response.content)
            except json.JSONDecodeError:
                # 如果失败，尝试使用正则表达式提取JSON部分
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    raise ValueError("无法从LLM响应中提取有效的JSON")
                params = json.loads(json_match.group(1))
            
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
                    "metadata": metadata
                }
            
            # 处理结果
            if isinstance(api_response, bytes):
                api_response = api_response.decode('utf-8')
            
            # 限制响应长度
            if len(api_response) > 10000:
                api_response = api_response[:10000] + "... [result is truncated]"
            
            # 保存esearch结果并进入下一步
            return {
                "messages": messages + [
                    AIMessage(
                        content=f"[{url}]->\n[{api_response}]",
                        additional_kwargs={"type": "eutils_progress", "parameters": params}
                    )
                ],
                "next": "fetch_details",  # 指向下一步
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"E-utilities esearch过程中出错: {str(e)}")
            return {
                "status": "error",
                "error": f"E-utilities esearch error: {str(e)}",
                "metadata": metadata
            }
    
    def fetch_details(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """第二步：获取详细信息，使用efetch或esummary API"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
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
                "metadata": metadata
            }

        # 获取第一条用户问题
        original_question = user_question[0].content
        logger.info(f"获取E-utilities详细信息: {original_question}")
        
        # 格式化历史记录为文本
        history_text = ""
        for msg in eutils_responses:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"
        
        # 构建单一提示
        combined_prompt = f"""
You are a parameter generator for NCBI E-utilities API. Based on the previous esearch results shown above, generate the parameters needed for an efetch or esummary request.

Extract the database IDs from the previous esearch results and use them for the next API call.
- Generally prefer esummary for better formatted data
- Use the same database (gene, snp, omim) as in the previous esearch call
- Include all relevant IDs found in the esearch results

Here are some examples:
Previous result contains IDs from gene database:
Output: {{"method": "efetch", "db": "gene", "id": "19171,5699,8138"}}

Previous result contains IDs from snp database:
Output: {{"method": "esummary", "db": "snp", "id": "1217074595"}}

Previous result contains IDs from omim database:
Output: {{"method": "esummary", "db": "omim", "id": "618767,601687,300778,148043,122100"}}

Only return a JSON object with these keys:
- method: Either "efetch" or "esummary" (prefer esummary when possible)
- db: The database to query (gene, snp, omim)
- id: Comma-separated list of IDs from the esearch results

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS HISTORY:
{history_text if history_text else "No previous interaction history."}
--------------------------------

Extract the IDs from the previous esearch result and include them in your JSON.
IMPORTANT: only return the JSON object, nothing else.
"""
        
        # 使用单一SystemMessage
        fetch_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(fetch_prompt)
            
            # 解析JSON响应
            try:
                # 尝试直接解析整个响应
                params = json.loads(response.content)
            except json.JSONDecodeError:
                # 如果失败，尝试使用正则表达式提取JSON部分
                json_match = re.search(r'({.*?})', response.content.replace('\n', ''))
                if not json_match:
                    raise ValueError("无法从LLM响应中提取有效的JSON")
                params = json.loads(json_match.group(1))
            
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
                    "metadata": metadata
                }
            
            # 处理结果
            if isinstance(api_response, bytes):
                api_response = api_response.decode('utf-8')
            
            # 限制响应长度
            if len(api_response) > 10000:
                api_response = api_response[:10000] + "... [result is truncated]"
            
            # 返回最终结果
            return {
                "messages": messages + [
                    AIMessage(
                        content=f"[{url}]->\n[{api_response}]",
                        additional_kwargs={"type": "eutils_response", "parameters": params}
                    )
                ],
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"E-utilities fetch过程中出错: {str(e)}")
            return {
                "status": "error",
                "error": f"E-utilities fetch error: {str(e)}",
                "metadata": metadata
            }

    def format_eutils_history(self, eutils_history: List[AIMessage]) -> str:
        return "\n".join([f"[{msg.content}]" for msg in eutils_history])