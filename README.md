# FusionRAG · 智能客服 Agent 系统

融合密集向量检索（FAISS）与稀疏关键词检索（BM25）的 RAG 智能客服系统，
支持 CrossEncoder 精排、LangChain Tool Calling、流式输出和多轮对话记忆。

## 技术架构

```
用户查询
  ├── FAISS 向量检索（bge-large-zh-v1.5）  → 语义相似 Top-10
  ├── BM25 关键词检索（jieba 分词）          → 词频匹配 Top-10
  ├── RRF 倒数排序融合                       → 去重合并候选集
  ├── CrossEncoder 精排（bge-reranker-large）→ 精细打分 Top-3
  └── Qwen LLM 生成回答（支持 Tool Calling）
```

## 项目结构

```
fusionrag-agent/
├── config.py              # 配置管理（模型、API、参数）
├── retriever.py           # 混合检索核心（FAISS + BM25 + CrossEncoder）
├── tools.py               # 自定义业务工具（订单/物流/退款）
├── agent.py               # Agent 核心（LLM + 工具 + 检索 + 记忆）
├── app.py                 # Streamlit Web 前端
├── cli.py                 # CLI 命令行入口
├── sample_knowledge.txt   # 示例知识库（电商 FAQ）
├── requirements.txt       # Python 依赖
└── README.md
```

## 快速启动

```bash
# 1. 安装依赖（Python 3.11 / 3.12）
pip install -r requirements.txt

# 2. 设置 DashScope API Key
export DASHSCOPE_API_KEY="sk-your-key-here"

# 3. 启动 CLI 模式
python cli.py

# 4. 或启动 Web 界面
streamlit run app.py
```

## CLI 命令

| 命令       | 功能             |
|-----------|-----------------|
| `/clear`  | 清空对话历史      |
| `/history`| 查看对话历史      |
| `/quit`   | 退出             |

## 核心模块说明

### 混合检索（retriever.py）
- **FAISS 向量检索**：用 bge-large-zh-v1.5 编码为 1024 维向量，IndexFlatIP 内积相似度
- **BM25 关键词检索**：jieba 中文分词 + BM25Okapi 算法
- **RRF 融合**：倒数排序融合，不依赖原始分数量纲
- **CrossEncoder 精排**：bge-reranker-large 对候选对做完整注意力打分

### Agent 工具调用（tools.py + agent.py）
- LangChain `@tool` 装饰器自动生成 JSON Schema
- `llm.bind_tools()` 将工具描述注入 LLM 上下文
- Agent 自主决策：知识问答走 RAG，操作类走工具调用

### 流式输出（agent.py → app.py）
- `ask_stream()` yield 两种事件：`("context", docs)` 和 `("token", text)`
- Streamlit 端逐 Token 实时渲染 + 光标效果
