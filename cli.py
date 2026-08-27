"""
FusionRAG CLI 命令行入口
==========================
终端交互模式，适合调试和快速验证。

启动命令：
    python cli.py                          # 使用默认知识库
    python cli.py --file my_docs.txt       # 指定知识库文件
    python cli.py --rebuild                # 强制重建索引

交互命令：
    /clear    清空对话历史
    /history  查看对话历史
    /quit     退出
"""

import argparse
import sys

from agent import FusionRAGAgent


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="FusionRAG 智能客服 CLI")
    parser.add_argument(
        "--file", type=str, default="sample_knowledge.txt",
        help="知识库文件路径（默认: sample_knowledge.txt）"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="强制重建索引（忽略缓存）"
    )
    args = parser.parse_args()

    # 初始化 Agent
    print("=" * 60)
    print("  FusionRAG · 智能客服 Agent (CLI)")
    print("  混合检索: FAISS + BM25 + CrossEncoder 精排")
    print("=" * 60)

    agent = FusionRAGAgent()

    # 加载知识库
    print(f"\n[启动] 加载知识库: {args.file}")
    agent.load_knowledge(args.file, force_rebuild=args.rebuild)

    print("\n[就绪] 输入问题开始对话，输入 /quit 退出")
    print("-" * 60)

    # 交互循环
    while True:
        try:
            query = input("\n👤 你: ").strip()

            if not query:
                continue

            # 内置命令
            if query.lower() == "/quit":
                print("再见！")
                break
            elif query.lower() == "/clear":
                agent.clear_history()
                print("[系统] 对话已清空")
                continue
            elif query.lower() == "/history":
                print(agent.get_history_summary())
                continue

            # 流式回答
            print("\n🤖 Agent: ", end="", flush=True)

            for event_type, data in agent.ask_stream(query):
                if event_type == "context":
                    # 在终端展示检索到的参考文档
                    print(f"\n   📚 [检索到 {len(data)} 条参考]")
                    for i, doc in enumerate(data, 1):
                        preview = doc.text[:80].replace('\n', ' ')
                        print(f"      [{i}] {preview}... (score={doc.score:.4f})")
                    print("   ", end="", flush=True)

                elif event_type == "token":
                    print(data, end="", flush=True)

            print()  # 换行

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
