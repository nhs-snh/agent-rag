"""
完整链路测试：DashScope Embedding + BM25 + FAISS + ReAct Agent
不依赖 sentence-transformers，无需下载本地模型
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ["DASHSCOPE_API_KEY"] = "sk-ws-H.EPPEXLE.Y5op.MEYCIQDnvKN1P54UFsdHkC705pAsRE7jNNwKReGkjGzA-8NeUQIhAKVDzLhPErgT0tf-JyDJV2JcxVRFO5-j-qrjKq8M23Lz"

print("=" * 60)
print("  FusionRAG 完整链路测试")
print("  DashScope Embedding + BM25 + FAISS + ReAct Agent")
print("=" * 60)

# 1. 测试知识库构建（DashScope Embedding API + FAISS + BM25）
print("\n[1/3] 构建知识库索引 ...")
from retriever import HybridRetriever
retriever = HybridRetriever()
retriever.build_index("sample_knowledge.txt", force_rebuild=True)

# 2. 测试混合检索
print("\n[2/3] 测试混合检索 ...")
query = "7天无理由退货运费谁出？"
print(f"  查询: {query}")
results = retriever.retrieve(query, top_k=3)
print(f"  检索到 {len(results)} 条结果:")
for i, doc in enumerate(results, 1):
    preview = doc.text[:80].replace('\n', ' ')
    print(f"    [{i}] score={doc.score:.4f} | {preview}...")

# 3. 测试完整 Agent（ReAct + RAG + 工具调用）
print("\n[3/3] 测试完整 Agent 问答 ...")
print("  问题: ORD-20240001 发货了吗？我想知道能不能退\n")

from agent import FusionRAGAgent
agent = FusionRAGAgent()
agent.load_knowledge("sample_knowledge.txt")
answer = agent.ask("ORD-20240001 发货了吗？我想知道能不能退")
print(f"\n  最终回答:\n  {answer}")

print("\n" + "=" * 60)
print("  [OK] 完整链路测试通过！")
print("=" * 60)
