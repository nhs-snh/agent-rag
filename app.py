"""
FusionRAG Streamlit 前端
==========================
Web 界面，支持：
- ReAct 推理过程可视化（Thought → Action → Observation 可视化）
- 流式 Token 输出（逐字渲染 LLM 回复）
- 检索上下文展示（展示 RAG 命中的参考文档）
- 多轮对话记忆（上下文连贯）
- 知识库文件上传（支持 .txt / .md / .pdf）
- 对话历史清空

启动命令：
    streamlit run app.py
"""

import streamlit as st
from agent import FusionRAGAgent
import os
import time

# 知识库上传目录（项目根目录下）
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_uploads")

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="FusionRAG · 智能客服 Agent",
    page_icon="🤖",
    layout="wide",
)

# ==================== 自定义样式 ====================

st.markdown("""
<style>
    /* 主标题 */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    /* 参考文档卡片 */
    .ref-card {
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    .ref-score {
        color: #667eea;
        font-weight: 600;
        font-size: 0.8rem;
    }
    /* ReAct 推理链样式 */
    .react-thought {
        background: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        font-style: italic;
        color: #856404;
    }
    .react-action {
        background: #d1ecf1;
        border-left: 3px solid #17a2b8;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #0c5460;
    }
    .react-observation {
        background: #d4edda;
        border-left: 3px solid #28a745;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Session State 初始化 ====================

@st.cache_resource
def init_agent():
    """
    初始化 Agent（全局单例）。
    @st.cache_resource 确保只初始化一次，多用户共享模型实例。
    """
    agent = FusionRAGAgent()
    return agent


def init_session():
    """初始化会话状态变量。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "knowledge_loaded" not in st.session_state:
        st.session_state.knowledge_loaded = False

    if "ref_docs" not in st.session_state:
        st.session_state.ref_docs = []

    # 当前使用的知识库文件路径
    if "current_kb_path" not in st.session_state:
        st.session_state.current_kb_path = ""


# ==================== 文件上传工具函数 ====================

def save_uploaded_file(uploaded_file) -> str:
    """
    保存 Streamlit 上传的文件到 knowledge_uploads/ 目录。
    返回保存后的绝对路径。

    PDF 文件会被提取纯文本后保存为 .txt，
    这样 retriever.py 的文本切片逻辑无需修改。
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 生成唯一文件名：时间戳 + 原始文件名，防止覆盖
    timestamp = int(time.time())
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        # PDF → 提取纯文本 → 保存为 .txt
        text = _extract_pdf_text(uploaded_file.read())
        save_name = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        # .txt / .md 直接保存原文
        save_name = f"{timestamp}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return save_path


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    从 PDF 二进制中提取纯文本。
    优先用 PyPDF2，没装则降级提示。
    """
    try:
        from PyPDF2 import PdfReader
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        return "（PDF 文本提取需要安装 PyPDF2：pip install PyPDF2）"


# ==================== 侧边栏 ====================

def render_sidebar(agent: FusionRAGAgent):
    """渲染侧边栏：知识库上传/加载、模型信息、对话管理。"""
    with st.sidebar:
        st.header("控制面板")

        # ---------- 知识库管理（Tab 切换：上传 / 路径） ----------
        st.subheader("知识库管理")

        tab_upload, tab_path = st.tabs(["上传文件", "指定路径"])

        with tab_upload:
            uploaded_file = st.file_uploader(
                "上传知识库文件",
                type=["txt", "md", "pdf"],
                help="支持 .txt / .md / .pdf 格式，PDF 会自动提取文本"
            )
            if uploaded_file is not None:
                # 显示文件信息
                size_kb = uploaded_file.size / 1024
                st.caption(f"{uploaded_file.name}  ({size_kb:.1f} KB)")

                if st.button("保存并构建索引", use_container_width=True):
                    with st.spinner("保存文件并构建索引中..."):
                        saved_path = save_uploaded_file(uploaded_file)
                        agent.load_knowledge(saved_path, force_rebuild=True)
                        st.session_state.knowledge_loaded = True
                        st.session_state.current_kb_path = saved_path
                    st.success("知识库加载成功")

        with tab_path:
            knowledge_file = st.text_input(
                "文件路径",
                value="sample_knowledge.txt",
                help="支持 .txt / .md 格式的文本文件，UTF-8 编码"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("加载知识库", use_container_width=True):
                    if os.path.exists(knowledge_file):
                        with st.spinner("构建索引中（首次约需2-5分钟）..."):
                            agent.load_knowledge(knowledge_file)
                            st.session_state.knowledge_loaded = True
                            st.session_state.current_kb_path = knowledge_file
                        st.success("知识库加载成功")
                    else:
                        st.error(f"文件不存在: {knowledge_file}")

            with col2:
                if st.button("强制重建", use_container_width=True):
                    target = st.session_state.current_kb_path or knowledge_file
                    if os.path.exists(target):
                        with st.spinner("重建索引中..."):
                            agent.load_knowledge(target, force_rebuild=True)
                        st.success("索引已重建")
                    else:
                        st.error(f"文件不存在: {target}")

        # 知识库状态
        status = "已加载" if st.session_state.knowledge_loaded else "未加载"
        kb_path = st.session_state.current_kb_path
        st.metric("知识库状态", status)
        if kb_path:
            st.caption(f"当前: {os.path.basename(kb_path)}")

        st.divider()

        # ---------- 已上传文件列表 ----------
        if os.path.exists(UPLOAD_DIR):
            files = sorted(os.listdir(UPLOAD_DIR), reverse=True)
            if files:
                with st.expander(f"已保存的知识库文件 ({len(files)})"):
                    for f in files[:10]:  # 最多展示 10 个
                        fpath = os.path.join(UPLOAD_DIR, f)
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.caption(f)
                        with col_b:
                            # 点击可快速切换为当前知识库
                            if st.button("加载", key=f"load_{f}", use_container_width=True):
                                with st.spinner("构建索引中..."):
                                    agent.load_knowledge(fpath, force_rebuild=True)
                                    st.session_state.knowledge_loaded = True
                                    st.session_state.current_kb_path = fpath
                                st.success(f"已加载: {f}")
                                st.rerun()

        st.divider()

        # ---------- 模型信息 ----------
        st.subheader("模型信息")
        st.text(f"LLM: qwen-plus (DashScope)")
        st.text(f"Embedding: text-embedding-v3")
        st.text(f"向量库: FAISS")
        st.text(f"稀疏检索: BM25 + jieba")

        st.divider()

        # ---------- 对话管理 ----------
        st.subheader("对话管理")
        if st.button("清空对话", use_container_width=True):
            agent.clear_history()
            st.session_state.messages = []
            st.session_state.ref_docs = []
            st.rerun()

        # 当前对话轮数
        msg_count = len(st.session_state.messages)
        st.metric("对话轮数", f"{msg_count // 2} 轮")


# ==================== 主界面 ====================

def render_main(agent: FusionRAGAgent):
    """渲染主界面：标题、对话消息、ReAct 推理链、输入框。"""

    # 标题
    st.markdown('<div class="main-title">FusionRAG · 智能客服 ReAct Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">ReAct 推理循环 · FAISS + BM25 混合检索 · CrossEncoder 精排 · Qwen · Tool Calling</div>',
        unsafe_allow_html=True,
    )

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent 流式回复
        with st.chat_message("assistant"):
            response_placeholder = st.empty()   # 最终回答占位
            trace_placeholder = st.empty()      # 推理链占位
            ref_placeholder = st.empty()         # 检索参考占位
            full_response = ""
            trace_html = ""                      # 推理链 HTML 拼接

            try:
                for event_type, data in agent.ask_stream(prompt):

                    if event_type == "context":
                        # 展示检索到的参考文档
                        st.session_state.ref_docs = data
                        with ref_placeholder.container():
                            st.markdown("**📚 检索参考：**")
                            for i, doc in enumerate(data, 1):
                                st.markdown(
                                    f'<div class="ref-card">'
                                    f'<span class="ref-score">参考{i} · 相关度 {doc.score:.4f}</span><br>'
                                    f'{doc.text[:200]}{"..." if len(doc.text) > 200 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    elif event_type == "thought":
                        # 💭 Agent 的推理过程（黄色卡片）
                        trace_html += (
                            f'<div class="react-thought">'
                            f'💭 <b>Thought:</b> {data}'
                            f'</div>'
                        )
                        trace_placeholder.markdown(trace_html, unsafe_allow_html=True)

                    elif event_type == "tool_call":
                        # 🔧 工具调用（蓝色卡片）
                        trace_html += (
                            f'<div class="react-action">'
                            f'🔧 <b>Action:</b> {data["name"]}'
                            f'({data["input"]}) — Step {data["step"]}'
                            f'</div>'
                        )
                        trace_placeholder.markdown(trace_html, unsafe_allow_html=True)

                    elif event_type == "observation":
                        # 👁️ 工具返回结果（绿色卡片）
                        preview = str(data)[:150]
                        trace_html += (
                            f'<div class="react-observation">'
                            f'👁️ <b>Observation:</b> {preview}'
                            f'{"..." if len(str(data)) > 150 else ""}'
                            f'</div>'
                        )
                        trace_placeholder.markdown(trace_html, unsafe_allow_html=True)

                    elif event_type == "token":
                        # 逐 Token 追加渲染最终回答
                        full_response += data
                        response_placeholder.markdown(full_response + "▌")

                # 流式结束，移除光标
                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ 回答出错: {str(e)}"
                response_placeholder.markdown(full_response)

            # 记录助手回复（包含推理链 + 最终回答）
            trace_text = trace_html if trace_html else ""
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })


# ==================== 入口 ====================

def main():
    init_session()
    agent = init_agent()
    render_sidebar(agent)
    render_main(agent)


if __name__ == "__main__":
    main()
