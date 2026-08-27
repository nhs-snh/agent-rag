"""
FusionRAG Agent 核心
=====================
将大模型、工具调用、RAG 检索、多轮记忆组装为完整的 Agent。

Agent 决策流程：
  用户提问
    ├── LLM 判断：需要调用工具？
    │   ├── 是 → 选择工具 → 执行 → 拿到结果 → 生成回答
    │   └── 否 → 走 RAG 检索 → 结合上下文生成回答
    └── 流式输出 Token

技术要点：
  - 使用 ChatOpenAI 兼容 DashScope 接口（base_url 指向 DashScope）
  - bind_tools() 将工具 JSON Schema 注入 LLM 上下文
  - 多轮对话用 ChatMessageHistory 维护
  - 流式输出用 stream() 方法逐 Token yield
"""

from typing import Generator, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

import config
from retriever import HybridRetriever
from tools import ALL_TOOLS


# ==================== Agent 系统提示词 ====================

SYSTEM_PROMPT = """你是 FusionRAG 智能客服助手，一个专业的电商客服 Agent。

你的能力：
1. **知识库问答**：基于检索到的知识库内容回答用户关于商品、政策、流程的问题。
2. **工具调用**：可以查询订单状态、物流信息、提交退款、查询退款政策。
3. **多轮对话**：记住之前的对话上下文，提供连贯的服务体验。

回答规范：
- 如果知识库中有答案，基于知识库内容回答，不要编造信息。
- 如果涉及订单/物流/退款操作，调用对应的工具获取实时数据。
- 如果知识库和工具都无法回答，诚实告知用户并建议联系人工客服。
- 回答要简洁、专业、有条理，适当使用列表让信息更清晰。
"""


# ==================== Agent 核心类 ====================

class FusionRAGAgent:
    """
    融合检索 + 工具调用的智能客服 Agent。

    使用方式：
        agent = FusionRAGAgent()
        agent.load_knowledge("sample_knowledge.txt")

        # 非流式
        answer = agent.ask("如何退货？")

        # 流式
        for event_type, data in agent.ask_stream("我的订单到哪了？"):
            if event_type == "context":
                print(f"检索到 {len(data)} 条参考")
            elif event_type == "token":
                print(data, end="", flush=True)
    """

    def __init__(self):
        # 初始化 LLM（DashScope 兼容 OpenAI 接口）
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.3,  # 客服场景用低温度，保证回答稳定性
            streaming=True,   # 启用流式输出
        )

        # 将工具的 JSON Schema 绑定到 LLM
        # bind_tools 后，LLM 的 prompt 中会包含工具描述
        # LLM 可以选择返回 tool_calls 来调用工具
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # 构建工具名 → 工具对象的映射，方便后续按名调用
        self.tool_map = {tool.name: tool for tool in ALL_TOOLS}

        # 混合检索器（延迟初始化，首次 load_knowledge 时构建）
        self.retriever = HybridRetriever()
        self._knowledge_loaded = False

        # 多轮对话记忆
        self.chat_history: BaseChatMessageHistory = InMemoryChatMessageHistory()

    # -------------------- 知识库管理 --------------------

    def load_knowledge(self, file_path: str, force_rebuild: bool = False):
        """
        加载知识库文件并构建索引。
        首次调用会下载嵌入模型和构建向量索引，约需 2-5 分钟。
        后续调用直接加载缓存，秒级启动。
        """
        print(f"[知识库] 加载文件: {file_path}")
        self.retriever.build_index(file_path, force_rebuild=force_rebuild)
        self._knowledge_loaded = True
        print("[知识库] 就绪")

    # -------------------- 工具调用执行 --------------------

    def _execute_tool_calls(self, ai_message) -> str:
        """
        执行 LLM 返回的工具调用请求。

        LLM 决定调用工具时，返回的 ai_message 包含 tool_calls 列表：
        [
            {"name": "query_order_status", "args": {"order_id": "ORD-xxx"}},
            ...
        ]

        本方法逐个执行工具，并将结果拼接为文本供 LLM 后续引用。
        """
        results = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in self.tool_map:
                print(f"[工具调用] {tool_name}({tool_args})")
                try:
                    result = self.tool_map[tool_name].invoke(tool_args)
                    results.append(f"[{tool_name}] {result}")
                except Exception as e:
                    results.append(f"[{tool_name}] 调用失败: {str(e)}")
            else:
                results.append(f"[{tool_name}] 未找到该工具")

        return "\n\n".join(results)

    # -------------------- 构建上下文 --------------------

    def _retrieve_context(self, query: str) -> str:
        """
        从知识库检索相关上下文，拼接为 prompt 的参考信息段。
        """
        if not self._knowledge_loaded:
            return ""

        docs = self.retriever.retrieve(query, top_k=config.RERANK_TOP_K)

        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[参考{i}]（相关度: {doc.score:.4f}）\n{doc.text}"
            )

        return "\n\n".join(context_parts)

    def _build_messages(self, query: str, context: str = "", tool_result: str = ""):
        """
        构建发送给 LLM 的完整消息列表。

        消息结构：
        [System] 系统提示词 + 知识库上下文
        [历史] 多轮对话记录
        [Human] 当前用户输入
        [Tool结果] 如果有工具调用结果
        """
        # System 消息：系统指令 + 知识库上下文
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\n以下是从知识库中检索到的参考信息，请基于这些信息回答用户问题：\n\n{context}"

        messages = [SystemMessage(content=system_content)]

        # 加入历史对话（多轮记忆）
        messages.extend(self.chat_history.messages)

        # 当前用户输入
        messages.append(HumanMessage(content=query))

        # 工具调用结果（如果有）
        if tool_result:
            messages.append(
                SystemMessage(content=f"以下是工具调用的返回结果：\n{tool_result}")
            )

        return messages

    # -------------------- 核心问答（非流式） --------------------

    def ask(self, query: str) -> str:
        """
        非流式问答：返回完整回答文本。

        流程：
        1. 检索知识库上下文
        2. 构建消息 → 调用 LLM
        3. 如果 LLM 返回了 tool_calls → 执行工具 → 再次调用 LLM
        4. 记录对话历史
        """
        # Step 1: 检索上下文
        context = self._retrieve_context(query)

        # Step 2: 首次调用 LLM（带工具绑定）
        messages = self._build_messages(query, context)
        response = self.llm_with_tools.invoke(messages)

        # Step 3: 如果 LLM 决定调用工具
        if response.tool_calls:
            tool_result = self._execute_tool_calls(response)

            # 用工具结果再次调用 LLM，让 LLM 基于工具返回的数据生成最终回答
            messages_with_tool = self._build_messages(query, context, tool_result)
            response = self.llm.invoke(messages_with_tool)

        # Step 4: 记录对话历史
        self.chat_history.add_message(HumanMessage(content=query))
        self.chat_history.add_message(AIMessage(content=response.content))

        return response.content

    # -------------------- 核心问答（流式） --------------------

    def ask_stream(self, query: str) -> Generator[Tuple[str, any], None, None]:
        """
        流式问答：逐 Token 产出回答。

        Yield 两种事件类型：
        - ("context", [doc1, doc2, ...])  → 检索到的上下文文档
        - ("token", "文本片段")             → LLM 生成的每个 Token

        使用方式：
            for event_type, data in agent.ask_stream("如何退货？"):
                if event_type == "context":
                    # 展示参考文档
                    for doc in data:
                        print(f"参考: {doc.text[:50]}...")
                elif event_type == "token":
                    # 逐字渲染
                    print(data, end="", flush=True)
        """
        # Step 1: 检索上下文
        context = self._retrieve_context(query)
        docs = self.retriever.retrieve(query, top_k=config.RERANK_TOP_K) if self._knowledge_loaded else []

        # 先 yield 检索结果（前端可展示"参考来源"）
        if docs:
            yield ("context", docs)

        # Step 2: 首次调用 LLM
        messages = self._build_messages(query, context)
        response = self.llm_with_tools.invoke(messages)

        # Step 3: 处理工具调用
        if response.tool_calls:
            tool_result = self._execute_tool_calls(response)

            # 带工具结果重新调用 LLM（流式）
            messages_with_tool = self._build_messages(query, context, tool_result)
            full_text = ""

            for chunk in self.llm.stream(messages_with_tool):
                if chunk.content:
                    full_text += chunk.content
                    yield ("token", chunk.content)

            # 记录历史
            self.chat_history.add_message(HumanMessage(content=query))
            self.chat_history.add_message(AIMessage(content=full_text))
            return

        # Step 4: 无工具调用，直接流式输出（需要重新调用，因为首次是非流式）
        # 注意：这里需要重新构建消息并流式调用
        full_text = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_text += chunk.content
                yield ("token", chunk.content)

        # 记录历史
        self.chat_history.add_message(HumanMessage(content=query))
        self.chat_history.add_message(AIMessage(content=full_text))

    # -------------------- 对话管理 --------------------

    def clear_history(self):
        """清空对话历史，开始新会话。"""
        self.chat_history.clear()
        print("[Agent] 对话历史已清空")

    def get_history_summary(self) -> str:
        """返回当前对话历史的摘要文本。"""
        messages = self.chat_history.messages
        if not messages:
            return "暂无对话历史"

        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}: {msg.content[:100]}...")
        return "\n".join(lines)
