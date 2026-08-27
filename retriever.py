"""
FusionRAG 混合检索引擎
=======================
核心架构：密集向量（FAISS） + 稀疏关键词（BM25 + jieba） → CrossEncoder 精排

检索流程：
  用户查询
    ├── FAISS 向量检索 → Top-10 语义相似文档
    ├── BM25 关键词检索 → Top-10 词频匹配文档
    ├── 去重合并（RRF 倒数排序融合）
    └── CrossEncoder 精排 → Top-3 最终结果

设计思路：
  - 向量检索擅长语义匹配（"退货"能匹配"退款""换货"）
  - BM25 擅长精确关键词匹配（订单号、产品型号等专有名词）
  - CrossEncoder 对合并后的候选集做精细打分，取最优
"""

import os
import pickle
from typing import List, Tuple

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ==================== 文档数据类 ====================

class Document:
    """
    轻量文档对象，存储文本块及其元数据。
    不依赖 LangChain 的 Document 类，保持检索模块独立。
    """
    def __init__(self, text: str, metadata: dict = None):
        self.text = text
        self.metadata = metadata or {}
        self.score = 0.0  # 检索得分

    def __repr__(self):
        preview = self.text[:60].replace('\n', ' ')
        return f"Document(score={self.score:.4f}, text='{preview}...')"


# ==================== 混合检索引擎 ====================

class HybridRetriever:
    """
    混合检索器：FAISS + BM25 + CrossEncoder 三级管线。

    使用方法：
        retriever = HybridRetriever()
        retriever.build_index("sample_knowledge.txt")  # 首次构建索引
        results = retriever.retrieve("如何退货？")        # 检索
    """

    def __init__(self):
        # 延迟加载模型，避免初始化时就下载大模型
        self._embedding_model = None
        self._reranker_model = None

        # 索引数据
        self._faiss_index = None       # FAISS 索引对象
        self._documents: List[Document] = []  # 全部文档块
        self._bm25 = None              # BM25 索引对象
        self._tokenized_corpus = []    # BM25 分词后的语料

    # -------------------- 模型懒加载 --------------------

    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        懒加载嵌入模型 bge-large-zh-v1.5。
        首次调用时从 HuggingFace 下载（约 1.2GB），之后走本地缓存。
        """
        if self._embedding_model is None:
            print(f"[加载模型] {config.EMBEDDING_MODEL} ...")
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def reranker_model(self) -> CrossEncoder:
        """
        懒加载 CrossEncoder 重排模型 bge-reranker-large。
        CrossEncoder 对每个 (query, doc) 对做完整注意力计算，精度高于 BiEncoder。
        """
        if self._reranker_model is None:
            print(f"[加载模型] {config.RERANKER_MODEL} ...")
            self._reranker_model = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker_model

    # -------------------- 文档预处理 --------------------

    def _split_documents(self, file_path: str) -> List[Document]:
        """
        读取文本文件并切片为文档块。

        切片策略：RecursiveCharacterTextSplitter
        - 优先按段落（\n\n）切分
        - 段落过长时按句子（。\n）切分
        - 句子仍超长时按字符硬切
        - chunk_overlap 保证相邻块上下文连续
        """
        # Windows 下先尝试 UTF-8，失败后 fallback GBK
        for encoding in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        # 使用 LangChain 的递归字符切片器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            # 按中文标点优先级切分，尽量保持句子完整
        )

        chunks = splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                text=chunk.strip(),
                metadata={"source": file_path, "chunk_index": i}
            )
            documents.append(doc)

        print(f"[切片完成] 共 {len(documents)} 个文档块")
        return documents

    # -------------------- FAISS 向量索引 --------------------

    def _build_faiss_index(self, documents: List[Document]):
        """
        构建 FAISS 向量索引。

        流程：
        1. 用 bge-large-zh-v1.5 将所有文档块编码为 1024 维向量
        2. 创建 FAISS IndexFlatIP（内积相似度，适合归一化向量）
        3. 将向量加入索引

        为什么用 IndexFlatIP 而非 IndexFlatL2？
        bge 模型输出的向量已做 L2 归一化，此时内积 = 余弦相似度。
        """
        import faiss

        print("[构建FAISS] 编码文档向量 ...")
        # normalize_embeddings=True 让向量 L2 归一化，内积等价于余弦相似度
        embeddings = self.embedding_model.encode(
            [doc.text for doc in documents],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # 创建内积索引（余弦相似度）
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self._faiss_index = index
        print(f"[FAISS完成] 索引维度={dimension}, 文档数={index.ntotal}")

    def _search_faiss(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        FAISS 向量检索：返回 (文档索引, 相似度分数) 列表。
        查询向量也做归一化，保证内积 = 余弦相似度 ∈ [0, 1]。
        """
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        query_vec = np.array(query_embedding, dtype=np.float32)

        scores, indices = self._faiss_index.search(query_vec, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    # -------------------- BM25 稀疏索引 --------------------

    def _build_bm25_index(self, documents: List[Document]):
        """
        构建 BM25 索引。

        BM25 核心思想：
        - TF（词频）：词在文档中出现越多，分越高，但有饱和效应
        - IDF（逆文档频率）：越稀有的词权重越高
        - 文档长度归一化：长文档不会仅因为词多就得分高

        jieba 分词将中文文本切为词列表，作为 BM25 的输入。
        """
        # jieba 分词：将每个文档块切为词列表
        self._tokenized_corpus = [
            list(jieba.cut(doc.text)) for doc in documents
        ]

        # 构建 BM25Okapi 索引（Okapi BM25 变体，最常用）
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"[BM25完成] 语料词数={len(self._bm25.idf)}")

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        BM25 关键词检索：对查询做 jieba 分词后计算 BM25 得分。
        返回 (文档索引, BM25分数) 列表，按分数降序。
        """
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top_k 个最高分
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # -------------------- RRF 融合 --------------------

    def _rrf_merge(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion（倒数排序融合）

        核心公式：RRF_score(d) = Σ 1 / (k + rank_i(d))
        - k 是平滑常数（默认60），防止排名靠前的文档得分过于集中
        - 对每个检索源的排名取倒数后求和
        - 不依赖原始分数的绝对值，只依赖排名，天然解决了
          FAISS 余弦相似度 和 BM25 分数 量纲不同的问题

        示例：某文档在 FAISS 排第1、BM25 排第3
          RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        """
        scores = {}

        # 向量检索排名
        for rank, (doc_idx, _) in enumerate(faiss_results):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        # BM25 检索排名
        for rank, (doc_idx, _) in enumerate(bm25_results):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        # 按 RRF 得分降序排列
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    # -------------------- CrossEncoder 精排 --------------------

    def _rerank(
        self, query: str, candidates: List[Tuple[int, float]], top_k: int
    ) -> List[Document]:
        """
        CrossEncoder 精排：对粗排结果做精细打分。

        与 BiEncoder（先编码再算相似度）不同：
        - CrossEncoder 将 query 和 document 拼接后一起输入模型
        - 模型内部做完整的 self-attention，能捕捉 query-doc 交互特征
        - 速度慢（不能预计算），但对 top-k 候选精排效果显著

        输入格式：[[query, doc_text], ...] → 每个 pair 输出一个相关度分数
        """
        if not candidates:
            return []

        # 构造 CrossEncoder 输入对
        pairs = []
        for doc_idx, _ in candidates:
            pairs.append([query, self._documents[doc_idx].text])

        # CrossEncoder 打分
        rerank_scores = self.reranker_model.predict(pairs)

        # 组装结果，按精排分数降序
        results = []
        for i, (doc_idx, _) in enumerate(candidates):
            doc = self._documents[doc_idx]
            doc.score = float(rerank_scores[i])
            results.append(doc)

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]

    # -------------------- 对外接口 --------------------

    def build_index(self, file_path: str, force_rebuild: bool = False):
        """
        构建完整索引（FAISS + BM25）。

        支持持久化：首次构建后保存到本地，下次直接加载，
        避免重复下载模型和计算向量。
        """
        os.makedirs(config.INDEX_DIR, exist_ok=True)

        # 如果有缓存且不需要强制重建，直接加载
        if (
            not force_rebuild
            and os.path.exists(config.FAISS_INDEX_PATH + ".faiss")
            and os.path.exists(config.BM25_CACHE_PATH)
        ):
            self._load_index()
            return

        # 文档切片
        self._documents = self._split_documents(file_path)

        # 构建双路索引
        self._build_faiss_index(self._documents)
        self._build_bm25_index(self._documents)

        # 持久化
        self._save_index()
        print("[索引持久化] 已保存到", config.INDEX_DIR)

    def _save_index(self):
        """将 FAISS 索引和 BM25 数据持久化到磁盘。"""
        import faiss

        faiss.write_index(self._faiss_index, config.FAISS_INDEX_PATH + ".faiss")

        # BM25 对象 + 文档块 + 分词语料一起序列化
        cache = {
            "bm25": self._bm25,
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(config.BM25_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    def _load_index(self):
        """从磁盘加载已持久化的索引。"""
        import faiss

        print("[加载索引] 从缓存恢复 ...")
        self._faiss_index = faiss.read_index(config.FAISS_INDEX_PATH + ".faiss")

        with open(config.BM25_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        self._bm25 = cache["bm25"]
        self._documents = cache["documents"]
        self._tokenized_corpus = cache["tokenized_corpus"]

        print(f"[加载完成] 文档数={len(self._documents)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        完整检索管线：双路召回 → RRF 融合 → CrossEncoder 精排。

        参数：
            query: 用户查询文本
            top_k: 最终返回的文档数（默认 config.RERANK_TOP_K）

        返回：
            按相关度降序排列的 Document 列表
        """
        top_k = top_k or config.RERANK_TOP_K

        # Step 1: 双路召回
        faiss_results = self._search_faiss(query, config.FAISS_TOP_K)
        bm25_results = self._search_bm25(query, config.BM25_TOP_K)

        # Step 2: RRF 融合，合并候选集并去重
        merged = self._rrf_merge(faiss_results, bm25_results)

        # 取融合后 Top-2K 候选送去精排（平衡精度和速度）
        rerank_candidates = merged[: config.RERANK_TOP_K * 3]

        # Step 3: CrossEncoder 精排
        final_results = self._rerank(query, rerank_candidates, top_k)

        return final_results
