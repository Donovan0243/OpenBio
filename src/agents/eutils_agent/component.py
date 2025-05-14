import re
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

You are a URL generator for NCBI E-utilities API. Generate the initial esearch URL for the NCBI E-utilities API.

DATABASE SELECTION RULES:
1. For gene names, symbols, aliases, or questions about gene function: use db=gene
2. For DNA sequences: DNA sequences should NOT be directly searched in E-utils! They require BLAST first.
3. For SNP (rs) IDs or questions about genetic variants: use db=snp
4. For genetic disorders, diseases, or phenotypes: use db=omim


Here are some examples:
Question: What is the official gene symbol of LMP10?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]->[......]
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&id=19171,5699,8138]->[......]

Question: Which gene is SNP rs1217074595 associated with?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&sort=relevance&id=1217074595]->[......]

Question: What are genes related to Meesmann corneal dystrophy?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy]->[......]
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&id=618767,601687,300778,148043,122100]->[......]


IMPORTANT GUIDELINES:
- If the question involves a DNA sequence, DO NOT use the sequence as a search term in E-utils
- For questions asking about aliases of genes containing DNA sequences, you should WAIT until BLAST results identify the gene
- Read the question carefully to determine what type of biological entity is being asked about

Only return the API URL in this format: [https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?retmax=10&db={{gene|snp|omim}}&retmode=json&sort=relevance&term={{term}}]

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS RESULTS:
{history_text if history_text else "No previous interaction history."}
--------------------------------

Extract relevant search terms from the user's question.
IMPORTANT: only return the API URL,put it in the [], no other text.
"""
        
        # 使用单一SystemMessage
        search_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(search_prompt)
            url_match = re.search(r'\[(https?://[^\[\]]+)\]', response.content)
            
            if not url_match:
                logger.error(f"无法从LLM响应中提取URL: {response}")
                return {
                    "messages": messages + [
                        AIMessage(content="Cannot generate valid E-utilities esearch URL",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
            
            url = url_match.group(1)
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
                        additional_kwargs={"type": "eutils_progress"}
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
You are a URL generator for NCBI E-utilities API. Based on the previous esearch results shown above, generate the efetch or esummary URL to get detailed information.

Only return the API URL in this format: [https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{{efetch|esummary}}.fcgi?retmax=10&db={{gene|snp|omim}}&retmode=json&id={{comma_separated_ids}}]

Extract the database IDs from the previous esearch results and use them in your efetch/esummary call.
- Generally prefer esummary for better formatted data
- Use the same database (gene, snp, omim) as in the previous esearch call
- Include all relevant IDs found in the esearch results

Here are some examples:
Question: What is the official gene symbol of LMP10?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]->[......]
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&id=19171,5699,8138]->[......]

Question: Which gene is SNP rs1217074595 associated with?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&sort=relevance&id=1217074595]->[......]

Question: What are genes related to Meesmann corneal dystrophy?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy]->[......]
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&id=618767,601687,300778,148043,122100]->[......]

--------------------------------
USER QUESTION:
{original_question}
--------------------------------
PREVIOUS RESULTS:
{history_text if history_text else "No previous E-utilities results."}
--------------------------------

Extract the IDs from the previous esearch result and include them in the id parameter.
IMPORTANT: only return the API URL,put it in the [], no other text.
"""
        
        # 使用单一SystemMessage
        fetch_prompt = [SystemMessage(content=combined_prompt)]
        
        try:
            response = self.llm.invoke(fetch_prompt)
            url_match = re.search(r'\[(https?://[^\[\]]+)\]', response.content)
            
            if not url_match:
                logger.error(f"无法从LLM响应中提取URL: {response}")
                return {
                    "messages": messages + [
                        AIMessage(content="Cannot generate valid E-utilities efetch/esummary URL",
                                additional_kwargs={"type": "eutils_error"})
                    ],
                    "status": "error",
                    "metadata": metadata
                }
            
            url = url_match.group(1)
            logger.info(f"生成的efetch/esummary URL: {url}")
            
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
                        additional_kwargs={"type": "eutils_response"}
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