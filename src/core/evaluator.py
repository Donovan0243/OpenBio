from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage,HumanMessage
import logging
from typing import Dict, Any
from textwrap import dedent

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

        self.prompt_template = """
        You are a strict answer evaluator.
        You should may a decision that whether the conversation history contain the answer.
        - If the answer is currently existing in the conversation history, output GENERATE
        - If the answer is not currently existing in the conversation history, output CONTINUE

        Here is some examples:
        Question: What is the official gene symbol of LMP10?
        Current conversation history:
        [https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]->
        [b'{"header":{"type":"esearch","version":"0.3"},"esearchresult":{"count":"3","retmax":"3","retstart":"0","idlist":["5699","8138","19171"],"translationset":[],"translationstack":[{"term":"LMP10[All Fields]","field":"All Fields","count":"3","explode":"N"},"GROUP"],"querytranslation":"LMP10[All Fields]"}}\n']
        Conclusion: CONTINUE
        Reason: Just got the id, have not got the gene symbol of LMP10.

        Question: What is the official gene symbol of LMP10?
        Current conversation history:
        [https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]->
        [b'{"header":{"type":"esearch","version":"0.3"},"esearchresult":{"count":"3","retmax":"3","retstart":"0","idlist":["5699","8138","19171"],"translationset":[],"translationstack":[{"term":"LMP10[All Fields]","field":"All Fields","count":"3","explode":"N"},"GROUP"],"querytranslation":"LMP10[All Fields]"}}\n']
        [https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&id=19171,5699,8138]->
        [b'\n1. Psmb10\nOfficial Symbol: Psmb10 and Name: proteasome (prosome, macropain) subunit, beta type 10 [Mus musculus (house mouse)]\nOther Aliases: Mecl-1, Mecl1\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; prosome Mecl1; proteasome (prosomome, macropain) subunit, beta type 10; proteasome MECl-1; proteasome subunit MECL1; proteasome subunit beta-2i\nChromosome: 8; Location: 8 53.06 cM\nAnnotation: Chromosome 8 NC_000074.7 (106662360..106665024, complement)\nID: 19171\n\n2. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: IMD121, LMP10, MECL1, PRAAS5, beta2i\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; proteasome (prosome, macropain) subunit, beta type, 10; proteasome MECl-1; proteasome catalytic subunit 2i; proteasome subunit MECL1; proteasome subunit beta 10; proteasome subunit beta 7i; proteasome subunit beta-2i; proteasome subunit beta2i\nChromosome: 16; Location: 16q22.1\nAnnotation: Chromosome 16 NC_000016.10 (67934506..67936850, complement)\nMIM: 176847\nID: 5699\n\n3. MECL1\nProteosome subunit MECL1 [Homo sapiens (human)]\nOther Aliases: LMP10, PSMB10\nThis record was replaced with GeneID: 5699\nID: 8138\n\n']
        Conclusion: GENERATE
        Reason: Official gene symbol (Psmb10) is found in the efetch result.
        
        """

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
        agent_history = [msg for msg in messages if isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") in ["blast_progress", "blast_response","eutils_progress", "eutils_response"]]

        # 格式化历史记录为文本
        history_text = ""
        for msg in agent_history:
            msg_type = msg.additional_kwargs.get("type", "unknown")
            history_text += f"\n--- {msg_type} ---\n{msg.content}\n"

        
        # 使用LLM评估信息是否足够

        eval_prompt = [
        SystemMessage(content=f"""\
{self.prompt_template2}
--------------------------------
User Question: {original_question}
--------------------------------
Current conversation history:
{history_text if history_text else "No previous conversation history."}
--------------------------------
Please return your decision DIRECTLY: GENERATE or CONTINUE
        """)
        ]
        
        response = self.llm.invoke(eval_prompt)
        
        logger.info(f"Evaluator决策: {response} (评估轮数: {eval_count}/5)")
        
        if "CONTINUE" in response.content:
            logger.info("信息不足，返回router继续查询")
            return {
                "next": "router",
                "metadata": state["metadata"]
            }
        elif "GENERATE" in response.content:
            logger.info("信息足够，进入生成阶段")
            return {
                "next": "generate",
                "metadata": state["metadata"]
            }
        else:
            logger.info("输出不规范，返回router继续查询")
            return {
                "next": "router",
                "metadata": state["metadata"]
            }
