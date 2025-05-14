<!--
 * @Author: Haodong Chen chd243013@gmail.com
 * @Date: 2025-05-07 23:36:52
 * @LastEditors: Haodong Chen chd243013@gmail.com
 * @LastEditTime: 2025-05-07 23:38:15
 * @FilePath: /OpenBioLLM-RAG/openbio_rag/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# OpenBio RAG

一个基于 LangChain 的 Agentic RAG 系统，用于生物信息学领域的智能问答。

## 功能特点

- 多 Agent 协作的智能问答系统
- 基于 LangChain 和 LangGraph 的工作流
- 支持网络搜索和知识库检索
- 灵活的决策和评估机制

## 安装

```bash
# 克隆项目
git clone [your-repo-url]
cd openbio_rag

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API keys
```

## 使用方法

```python
from src.graph.workflow import RAGWorkflow

# 创建工作流
workflow = RAGWorkflow()
app = workflow.create_graph()

# 运行查询
result = app.invoke({
    "messages": ["你的问题"],
    "next_step": "decide",
    "action_history": [],
    "current_result": "",
    "attempts": 0
})
```

## 项目结构

```
openbio_rag/
├── notebooks/          # Jupyter notebooks
├── src/               # 源代码
│   ├── agents/        # Agent 实现
│   ├── types/         # 类型定义
│   ├── tools/         # 工具集合
│   ├── prompts/       # 提示词模板
│   └── graph/         # 工作流定义
├── tests/             # 测试
└── config/            # 配置文件
```

## 开发

1. 创建虚拟环境
2. 安装开发依赖
3. 运行测试

## License

MIT
