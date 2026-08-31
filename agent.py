"""
FusionRAG ReAct Agent 核心
============================
基于 ReAct（Reasoning + Acting）模式的智能客服 Agent。

与之前 Tool Calling 版本的核心区别：
┌─────────────────────┬────────────────────┬──────────────────────────┐
│                     │ Tool Calling 版    │ ReAct Agent 版           │
├─────────────────────┼────────────────────┼──────────────────────────┤
│ 思考过程            │ 无（黑盒决策）      │ 每步输出推理链（Thought） │
│ 多步推理            │ 工具调完直接回答    │ Think→Act→Observe 循环   │
│ 自我纠错            │ 工具失败就结束      │ 观察失败原因，换策略重试  │
│ 任务规划            │ 无                 │ 复杂任务先拆解步骤        │
│ 终止条件            │ 没有 tool_calls     │ 显式判断"信息是否足够"    │
└─────────────────────┴────────────────────┴──────────────────────────┘

ReAct 循环流程：
  用户提问
    │
    ├── [Plan] 复杂任务？先拆解为步骤列表
    │
    └── 循环（最多 max_iterations 轮）：
          │
          ├── Thought: 我目前知道什么？还需要什么信息？
          ├── Action:  决定调用哪个工具，传什么参数
          ├── Observation: 工具返回了什么结果
          │
          ├── 信息够了？→ 生成最终回答（Final Answer）
          └── 信息不够？→ 继续下一轮循环
"""

from typing import Generator, Tuple, List
import json
import re
import time
import random

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

import config
from retriever import HybridRetriever
from tools import ALL_TOOLS


# ==================== ReAct 系统提示词 ====================

REACT_SYSTEM_PROMPT = """你是 FusionRAG 智能客服 Agent，基于 ReAct（推理+行动）模式工作。

## 你的工作方式

每一步你必须按以下格式思考和行动：

Thought: [你当前的推理过程：已知什么、缺什么、下一步打算做什么]
Action: [工具名称]
Action Input: [JSON 格式的工具参数]

当你收集到足够信息可以回答用户时，输出：

Thought: [总结推理过程，说明为什么现在可以回答了]
Final Answer: [给用户的最终回答]

## 可用工具

{tool_descriptions}

## 知识库参考

以下是从知识库中检索到的相关内容（如果有）：

{context}

## 重要规则

1. **必须先 Thought 再 Action**：每次调用工具前，先说明你为什么需要这个信息。
2. **工具失败时换策略**：如果一个工具调用失败或返回空结果，不要重复同样的调用，换一个工具或换参数重试。
3. **基于事实回答**：回答必须基于工具返回的真实数据或知识库内容，不要编造。
4. **不要重复调用**：如果一个工具已经返回了你需要的信息，不要再调用一次。
5. **最多 {max_iterations} 步**：如果超过最大步数仍无法回答，诚实告知用户。
"""


# ==================== 工具描述生成 ====================

def _build_tool_descriptions() -> str:
    """
    将所有工具的 JSON Schema 转为人类可读的描述文本。
    这比直接传 JSON 更适合 ReAct 模式的 prompt 注入。
    """
    descriptions = []
    for tool in ALL_TOOLS:
        desc = f"- **{tool.name}**: {tool.description}"

        # 从工具的 args_schema 提取参数说明
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema()
            props = schema.get("properties", {})
            required = schema.get("required", [])

            param_lines = []
            for param_name, param_info in props.items():
                req = "(必填)" if param_name in required else "(可选)"
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                param_lines.append(
                    f"    - {param_name} ({param_type}) {req}: {param_desc}"
                )

            if param_lines:
                desc += "\n  参数:\n" + "\n".join(param_lines)

        descriptions.append(desc)

    return "\n\n".join(descriptions)


# ==================== ReAct 解析器 ====================

class ReActParser:
    """
    解析 LLM 输出的 ReAct 格式。

    LLM 可能输出三种格式：
    1. Thought + Action + Action Input → 需要调用工具
    2. Thought + Final Answer → 生成最终回答
    3. 纯文本（格式不标准）→ 降级为直接回答

    解析策略：
    - 用正则匹配 Thought / Action / Action Input / Final Answer
    - Action Input 必须是 JSON 格式
    - 如果都没匹配到，降级为纯文本回答
    """

    ACTION_PATTERN = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    ACTION_INPUT_PATTERN = re.compile(
        r"Action Input:\s*(\{.*?\})", re.IGNORECASE | re.DOTALL
    )
    FINAL_ANSWER_PATTERN = re.compile(
        r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL
    )
    THOUGHT_PATTERN = re.compile(
        r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)",
        re.IGNORECASE | re.DOTALL
    )

    @staticmethod
    def parse(llm_output: str) -> dict:
        """
        解析 LLM 输出，返回结构化结果。

        返回格式：
        {
            "type": "action" | "final" | "plain",
            "thought": "...",
            "action": "tool_name",       # 仅 type=action
            "action_input": {...},        # 仅 type=action
            "final_answer": "...",        # 仅 type=final
        }
        """
        # 优先级1：Final Answer
        final_match = ReActParser.FINAL_ANSWER_PATTERN.search(llm_output)
        if final_match:
            thought_match = ReActParser.THOUGHT_PATTERN.search(llm_output)
            return {
                "type": "final",
                "thought": thought_match.group(1).strip() if thought_match else "",
                "final_answer": final_match.group(1).strip(),
            }

        # 优先级2：Action（需要调用工具）
        action_match = ReActParser.ACTION_PATTERN.search(llm_output)
        if action_match:
            thought_match = ReActParser.THOUGHT_PATTERN.search(llm_output)

            action_input = {}
            action_input_match = ReActParser.ACTION_INPUT_PATTERN.search(llm_output)
            if action_input_match:
                try:
                    action_input = json.loads(action_input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {"raw_input": action_input_match.group(1).strip()}

            return {
                "type": "action",
                "thought": thought_match.group(1).strip() if thought_match else "",
                "action": action_match.group(1).strip(),
                "action_input": action_input,
            }

        # 优先级3：降级为纯文本
        return {
            "type": "plain",
            "thought": "",
            "final_answer": llm_output.strip(),
        }


# ==================== ReAct Agent 核心类 ====================

class FusionRAGAgent:
    """
    ReAct 模式的智能客服 Agent。

    核心循环（_react_step 方法）：
        1. 构建消息 → 调用 LLM
        2. 解析输出 → 判断类型（Action / Final Answer / Plain）
        3. 如果是 Action → 执行工具 → 把 Observation 加入消息历史 → 回到1
        4. 如果是 Final Answer → 结束循环
        5. 最多循环 max_iterations 轮

    与 Tool Calling 版的关键区别：
        - Tool Calling 版：LLM 返回结构化的 tool_calls，框架自动解析执行
        - ReAct 版：LLM 输出自然语言的 Thought/Action，框架用正则解析
        - ReAct 版的优势：推理过程可见、支持自我纠错、可解释性强
    """

    def __init__(self, max_iterations: int = 6):
        # LLM（不开 streaming，因为需要完整解析 ReAct 格式）
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.3,
            streaming=False,
        )

        # 流式 LLM（仅用于最终回答的流式输出）
        self.llm_streaming = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.3,
            streaming=True,
        )

        # 工具映射
        self.tool_map = {tool.name: tool for tool in ALL_TOOLS}
        self.tool_descriptions = _build_tool_descriptions()

        # 混合检索器
        self.retriever = HybridRetriever()
        self._knowledge_loaded = False

        # 多轮对话记忆
        self.chat_history: BaseChatMessageHistory = InMemoryChatMessageHistory()

        # Agent 安全参数：最大推理步数（防止无限循环）
        self.max_iterations = max_iterations

        # LLM 重试参数
        self.max_retries = 3          # 最大重试次数
        self.retry_base_delay = 1.0   # 基础退避时间（秒）

    # -------------------- LLM 调用重试 --------------------

    def _call_llm_with_retry(self, messages):
        """
        带指数退避重试的 LLM 调用（非流式）。

        重试策略：
        - 指数退避：delay = base_delay * 2^attempt + random_jitter
        - 第1次重试等 ~1s，第2次 ~2s，第3次 ~4s
        - 随机抖动（0~1s）防止多个请求同时重试（惊群效应）
        - 可重试的错误：网络超时、429限流、500/502/503服务端故障
        - 不可重试的错误：400参数错误、401认证失败（重试也没用）
        """
        for attempt in range(self.max_retries):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                error_str = str(e).lower()

                # 不可重试的错误：认证失败、参数错误
                if any(code in error_str for code in ["401", "400", "invalid api key", "authentication"]):
                    raise

                # 最后一次重试也失败了
                if attempt == self.max_retries - 1:
                    print(f"[LLM重试] 已重试 {self.max_retries} 次仍失败: {e}")
                    raise

                # 指数退避 + 随机抖动
                delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[LLM重试] 第{attempt+1}次失败: {type(e).__name__}, "
                      f"{delay:.1f}s 后重试...")
                time.sleep(delay)

    def _stream_llm_with_retry(self, messages):
        """
        带指数退避重试的 LLM 流式调用。

        与 _call_llm_with_retry 逻辑相同，但调用 stream() 而非 invoke()。
        返回的是 generator，逐 chunk yield 给调用方。
        """
        for attempt in range(self.max_retries):
            try:
                for chunk in self.llm_streaming.stream(messages):
                    yield chunk
                return  # 流式正常结束
            except Exception as e:
                error_str = str(e).lower()

                if any(code in error_str for code in ["401", "400", "invalid api key", "authentication"]):
                    raise

                if attempt == self.max_retries - 1:
                    print(f"[LLM重试] 流式调用已重试 {self.max_retries} 次仍失败: {e}")
                    raise

                delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[LLM重试] 流式第{attempt+1}次失败: {type(e).__name__}, "
                      f"{delay:.1f}s 后重试...")
                time.sleep(delay)

    # -------------------- 知识库管理 --------------------

    def load_knowledge(self, file_path: str, force_rebuild: bool = False):
        """加载知识库并构建索引。"""
        print(f"[知识库] 加载文件: {file_path}")
        self.retriever.build_index(file_path, force_rebuild=force_rebuild)
        self._knowledge_loaded = True
        print("[知识库] 就绪")

    # -------------------- 上下文检索 --------------------

    def _retrieve_context(self, query: str) -> str:
        """从知识库检索上下文。"""
        if not self._knowledge_loaded:
            return "（知识库未加载）"

        docs = self.retriever.retrieve(query, top_k=config.RERANK_TOP_K)
        if not docs:
            return "（未检索到相关内容）"

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[参考{i}]（相关度: {doc.score:.4f}）\n{doc.text}")
        return "\n\n".join(parts)

    # -------------------- 消息构建 --------------------

    def _build_react_messages(
        self, query: str, context: str, react_trace: List[str]
    ) -> List:
        """
        构建 ReAct 循环的消息列表。

        消息结构：
        [System] ReAct 提示词 + 工具描述 + 知识库上下文
        [历史]   多轮对话记录（压缩为摘要，避免 prompt 过长）
        [Human]  当前用户输入
        [Trace]  本轮 ReAct 的 Thought/Action/Observation 推理痕迹
        """
        system_content = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions,
            context=context,
            max_iterations=self.max_iterations,
        )

        messages = [SystemMessage(content=system_content)]

        # 加入历史对话（只保留最近 6 条，避免 prompt 过长）
        recent_history = self.chat_history.messages[-6:]
        messages.extend(recent_history)

        # 当前用户输入
        messages.append(HumanMessage(content=query))

        # 加入本轮 ReAct 的推理痕迹
        if react_trace:
            trace_text = "\n\n".join(react_trace)
            messages.append(AIMessage(content=trace_text))

        return messages

    # -------------------- 工具执行 --------------------

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        执行工具调用，包含错误处理和降级。

        错误处理策略（让 LLM 在下一轮 Thought 中看到错误并自主调整）：
        1. 工具不存在 → 返回可用工具列表
        2. 参数错误 → 返回错误详情
        3. 执行异常 → 返回异常信息
        """
        if tool_name not in self.tool_map:
            available = ", ".join(self.tool_map.keys())
            return f"错误：工具 '{tool_name}' 不存在。可用工具：{available}"

        try:
            result = self.tool_map[tool_name].invoke(tool_input)
            return str(result)
        except TypeError as e:
            return f"参数错误：{str(e)}。请检查 Action Input 的参数名和类型。"
        except Exception as e:
            return f"工具执行异常：{str(e)}。请尝试换一种方式或换一个工具。"

    # -------------------- ReAct 核心循环 --------------------

    def _react_step(self, query: str, context: str) -> Generator:
        """
        执行完整的 ReAct 循环。这是 Agent 的核心方法。

        每一轮循环：
        1. 构建消息（包含之前所有 Thought/Action/Observation）
        2. 调用 LLM 获取下一步决策
        3. 解析 LLM 输出：
           - Final Answer → 结束，yield 最终回答
           - Action → 执行工具，把 Observation 加入 trace，继续下一轮
           - Plain → 降级，直接作为回答

        Yield 事件类型：
        - ("context", docs)        → 检索到的知识库文档
        - ("thought", text)        → Agent 的推理过程（可解释性）
        - ("tool_call", info)      → 工具调用信息 {name, input, step}
        - ("observation", text)    → 工具返回结果
        - ("final", text)          → 最终回答
        - ("error", text)          → 超时/异常信息
        """
        # Step 0: 知识库检索
        docs = (
            self.retriever.retrieve(query, top_k=config.RERANK_TOP_K)
            if self._knowledge_loaded else []
        )
        if docs:
            yield ("context", docs)

        react_trace = []  # 本轮所有 Thought/Action/Observation

        for step in range(self.max_iterations):
            # Step 1: 构建消息并调用 LLM（带指数退避重试）
            messages = self._build_react_messages(query, context, react_trace)
            response = self._call_llm_with_retry(messages)
            llm_output = response.content.strip()

            # Step 2: 解析 ReAct 格式
            parsed = ReActParser.parse(llm_output)

            # Step 3a: Final Answer → 结束
            if parsed["type"] == "final":
                if parsed["thought"]:
                    yield ("thought", parsed["thought"])
                yield ("final", parsed["final_answer"])
                return

            # Step 3b: Action → 执行工具，继续循环
            elif parsed["type"] == "action":
                if parsed["thought"]:
                    yield ("thought", parsed["thought"])

                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                # yield 工具调用事件（前端可展示"正在查询..."）
                yield ("tool_call", {
                    "name": tool_name,
                    "input": tool_input,
                    "step": step + 1,
                })

                # 执行工具
                observation = self._execute_tool(tool_name, tool_input)

                # yield 观察结果
                yield ("observation", observation)

                # 记录推理痕迹（下一轮 LLM 能看到之前所有步骤）
                action_block = (
                    f"Thought: {parsed['thought']}\n"
                    f"Action: {tool_name}\n"
                    f"Action Input: {json.dumps(tool_input, ensure_ascii=False)}\n"
                    f"Observation: {observation}"
                )
                react_trace.append(action_block)

            # Step 3c: Plain → 降级处理
            elif parsed["type"] == "plain":
                yield ("final", parsed["final_answer"])
                return

        # 超过最大迭代次数
        yield ("error", f"Agent 在 {self.max_iterations} 步内未能得出结论，请尝试简化问题。")

    # -------------------- 对外接口 --------------------

    def ask(self, query: str) -> str:
        """非流式问答。"""
        context = self._retrieve_context(query)
        final_answer = ""

        for event_type, data in self._react_step(query, context):
            if event_type == "final":
                final_answer = data
            elif event_type == "error":
                final_answer = f"⚠️ {data}"

        # 记录对话历史
        self.chat_history.add_message(HumanMessage(content=query))
        self.chat_history.add_message(AIMessage(content=final_answer))

        return final_answer

    def ask_stream(self, query: str) -> Generator[Tuple[str, any], None, None]:
        """
        流式问答。

        ReAct 的思考过程（Thought/Action/Observation）是非流式的，
        因为需要完整文本才能解析 ReAct 格式。
        只有最终回答（Final Answer）阶段切换为流式输出。

        Yield 事件类型：
        - ("context", docs)      → 检索结果
        - ("thought", text)      → 推理过程（可解释性，前端可折叠展示）
        - ("tool_call", info)    → 工具调用（前端可展示"正在查询..."）
        - ("observation", text)  → 工具结果（前端可展示中间结果）
        - ("token", text)        → 最终回答的每个 Token（逐字渲染）
        """
        context = self._retrieve_context(query)
        full_response = ""

        for event_type, data in self._react_step(query, context):

            # 思考过程事件：直接透传
            if event_type in ("context", "thought", "tool_call", "observation"):
                yield (event_type, data)

            # 最终回答：切换为流式 LLM 逐 Token 输出
            elif event_type == "final":
                stream_messages = [
                    SystemMessage(content=(
                        "你是 FusionRAG 智能客服。以下是你经过推理得出的结论，"
                        "请直接、自然地输出给用户的回答，不要重复推理过程：\n\n"
                        f"{data}"
                    ))
                ]

                for chunk in self._stream_llm_with_retry(stream_messages):
                    if chunk.content:
                        full_response += chunk.content
                        yield ("token", chunk.content)

            elif event_type == "error":
                error_text = f"⚠️ {data}"
                yield ("token", error_text)
                full_response = error_text

        # 记录对话历史
        if full_response:
            self.chat_history.add_message(HumanMessage(content=query))
            self.chat_history.add_message(AIMessage(content=full_response))

    # -------------------- 对话管理 --------------------

    def clear_history(self):
        """清空对话历史。"""
        self.chat_history.clear()
        print("[Agent] 对话历史已清空")

    def get_history_summary(self) -> str:
        """返回对话历史摘要。"""
        messages = self.chat_history.messages
        if not messages:
            return "暂无对话历史"

        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}: {msg.content[:100]}...")
        return "\n".join(lines)
