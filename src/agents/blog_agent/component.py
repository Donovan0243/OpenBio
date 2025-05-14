from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

class WebAgent:
    def __init__(self):
        # 初始化基础组件
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOllama(
            model="qwen2.5:3b",
            base_url="http://34.142.153.30:11434",
            api_key="ollama"
        )

                # 加载和分割文档
        self.urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        
        self._load_and_process_documents()
        
        # 创建检索工具
        self.retriever = self.vectorstore.as_retriever()
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
        )
        
    def _load_and_process_documents(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        # 存储文档
        self.vectorstore.add_documents(doc_splits)

    def process_query(self, state):
        messages = state["messages"]
        question = messages[0].content
        
        # 使用检索器获取相关文档
        retriever = self.vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 生成答案
        prompt = ChatPromptTemplate.from_template("""
        基于以下上下文回答问题:
        --------------------
        {context}
        --------------------
        问题: {question}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return {"messages": [response]}

    def evaluate_documents(self, state):
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        # 评估相关性
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "context": docs})
        
        if "yes" in result.lower():
            return "generate"
        return "rewrite"

    def rewrite(self, state):
        messages = state["messages"]
        question = messages[0].content
        
        prompt = ChatPromptTemplate.from_template("""
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question:
        {question}
        Formulate an improved question:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})
        return {"messages": [response]}

    def generate(self, state):
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content

        prompt = hub.pull("rlm/rag-prompt")
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
