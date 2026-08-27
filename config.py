"""
FusionRAG 配置管理
==================
集中管理所有模型路径、API Key、检索参数等配置项。
DashScope 兼容 OpenAI 接口，所以用 base_url 指向 DashScope 端点。
"""

import os

# ==================== 大模型配置 ====================
# 通义千问 DashScope API（兼容 OpenAI 接口格式）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-your-api-key-here")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"  # 可选: qwen-plus / qwen-turbo / qwen-max

# ==================== 嵌入模型配置 ====================
# bge-large-zh-v1.5：BAAI 出品，中文语义检索 SOTA 级模型
# 输出维度 1024，适合 FAISS 索引
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DIMENSION = 1024

# ==================== 重排模型配置 ====================
# bge-reranker-large：CrossEncoder 架构，对候选文档做精细打分
# 比 BiEncoder 慢但准确率高，适合 Top-K 精排
RERANKER_MODEL = "BAAI/bge-reranker-large"

# ==================== 检索参数 ====================
# 向量检索返回 Top-K 候选数
FAISS_TOP_K = 10

# BM25 检索返回 Top-K 候选数
BM25_TOP_K = 10

# CrossEncoder 重排后最终保留的文档数
RERANK_TOP_K = 3

# 混合检索时向量/BM25的权重（向量权重 + BM25权重 = 1.0）
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4

# ==================== 文档切片参数 ====================
CHUNK_SIZE = 500        # 每块最大字符数
CHUNK_OVERLAP = 50      # 相邻块重叠字符数

# ==================== 持久化路径 ====================
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index_data")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")
BM25_CACHE_PATH = os.path.join(INDEX_DIR, "bm25_cache.pkl")
