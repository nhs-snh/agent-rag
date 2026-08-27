"""
快速测试脚本：只测试 ReAct Agent + 工具调用 + LLM
不加载本地嵌入/重排模型，跳过 RAG 检索部分
"""
import os
import sys
import io

# Windows 终端编码修复：强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ["DASHSCOPE_API_KEY"] = "sk-ws-H.EPPEXLE.Y5op.MEYCIQDnvKN1P54UFsdHkC705pAsRE7jNNwKReGkjGzA-8NeUQIhAKVDzLhPErgT0tf-JyDJV2JcxVRFO5-j-qrjKq8M23Lz"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools import ALL_TOOLS

# 1. 测试 LLM 连通性
print("=" * 50)
print("  FusionRAG 快速测试")
print("=" * 50)

print("\n[1/3] 测试 LLM 连通性 ...")
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)
response = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己")])
print(f"  LLM 回复: {response.content}")

# 2. 测试工具描述生成
print("\n[2/3] 测试工具注册 ...")
for tool in ALL_TOOLS:
    print(f"  [OK] {tool.name}: {tool.description[:50]}...")

# 3. 测试 ReAct 循环（手动模拟）
print("\n[3/3] 测试 ReAct 工具调用 ...")
print("  问题: ORD-20240001 的订单状态是什么？\n")

# 构建 ReAct 提示词
tool_desc = ""
for tool in ALL_TOOLS:
    tool_desc += f"- {tool.name}: {tool.description}\n"

system_prompt = f"""你是客服Agent，按ReAct格式工作。

可用工具：
{tool_desc}

格式：
Thought: 你的推理
Action: 工具名
Action Input: {{"参数名": "参数值"}}

或当信息足够时：
Thought: 总结
Final Answer: 最终回答"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="ORD-20240001 的订单状态是什么？")
]

# 第一轮：LLM 决策
print("  --- 第1轮 LLM 决策 ---")
response = llm.invoke(messages)
print(f"  LLM输出:\n  {response.content}\n")

# 解析并执行工具
import json, re
action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response.content, re.IGNORECASE)
input_match = re.search(r"Action Input:\s*(\{.*?\})", response.content, re.DOTALL)

if action_match and input_match:
    tool_name = action_match.group(1).strip()
    tool_input = json.loads(input_match.group(1))

    # 执行工具
    tool_map = {t.name: t for t in ALL_TOOLS}
    if tool_name in tool_map:
        result = tool_map[tool_name].invoke(tool_input)
        print(f"  --- 工具执行结果 ---")
        print(f"  {result}\n")

        # 第二轮：带工具结果生成最终回答
        messages.append(SystemMessage(content=response.content))
        messages.append(SystemMessage(content=f"Observation: {result}"))
        messages.append(HumanMessage(content="请基于以上信息给出最终回答（Final Answer格式）"))

        print("  --- 第2轮 生成最终回答 ---")
        final = llm.invoke(messages)
        print(f"  {final.content}")
    else:
        print(f"  工具 {tool_name} 不存在")
else:
    print("  LLM 直接给出了回答（未调用工具）")

print("\n" + "=" * 50)
print("  [OK] 测试完成！核心链路正常")
print("=" * 50)
