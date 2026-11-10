"""CLI hỏi đáp end-to-end với components THẬT (LLM từ .env + Milvus + Neo4j).

    python -m agent.cli "bé sốt 3 ngày nên làm gì?"        # 1 câu
    python -m agent.cli --chat                             # chat nhiều lượt (conversation)
    python -m agent.cli --trace "paracetamol là thuốc gì"   # kèm trace + tool results

Chế độ --chat: giữ history đa lượt — câu follow-up ("còn ho?") hiểu ngữ cảnh
câu trước. Lệnh trong chat: /reset (xóa lịch sử), /exit hoặc Ctrl-C (thoát).

Cần: .env có LLM_API_KEY (DeepSeek hoặc Gemini — xem .env.example),
docker compose up -d, đã chạy make index (vector) / make kg (KG, optional).
"""
from __future__ import annotations

import argparse
import json
import sys


def _print_answer(answer, *, show_trace: bool = False) -> None:
    print(f"[{answer.category}] (attempts={answer.attempts}, trace_id={answer.trace_id})")
    print(answer.text)
    if answer.citations:
        print("\ncitations:")
        for c in answer.citations:
            print(f"  - {c.source_row_id}#{c.chunk_index}: {c.snippet}")
    if show_trace:
        print("\ntrace:")
        for line in answer.trace:
            print(f"  {line}")
        if answer.tool_results:
            print("\ntool_results:")
            print(json.dumps(answer.tool_results, ensure_ascii=False, indent=2))
        if answer.dropped_sources:
            print(f"\ndropped_sources: {answer.dropped_sources}")


def _chat(graph, show_trace: bool) -> int:
    """Vòng lặp chat nhiều lượt — giữ history cho câu follow-up."""
    history: list[dict] = []
    print("chat mode — /reset: xóa lịch sử, /exit hoặc Ctrl-C: thoát")
    while True:
        try:
            question = input("\nbạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/exit", "/quit"):
            return 0
        if question == "/reset":
            history.clear()
            print("(đã xóa lịch sử)")
            continue
        try:
            answer = graph.ask(question, history=history)
        except KeyboardInterrupt:
            print("\n(hủy câu hỏi — gõ /exit để thoát)")
            continue
        except Exception as e:  # noqa: BLE001 — API lỗi thì báo và hỏi tiếp, không kill phiên
            print(f"[lỗi] {type(e).__name__}: {e}")
            print("(kiểm tra .env / mạng / quota API rồi hỏi tiếp — /exit để thoát)")
            continue
        # chỉ đưa vào history câu trả lời sạch (trước disclaimer) + câu hỏi đã qua gate
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer.raw_text or answer.text})
        _print_answer(answer, show_trace=show_trace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hỏi đáp qua full agent graph (components thật)")
    parser.add_argument("question", nargs="?", help="câu hỏi tiếng Việt (bỏ trống nếu dùng --chat)")
    parser.add_argument("--chat", action="store_true", help="chat nhiều lượt (conversation)")
    parser.add_argument("--trace", action="store_true", help="in trace + tool_results (debug)")
    parser.add_argument("--json", action="store_true", help="in output JSON (dùng cho scripting)")
    args = parser.parse_args(argv)

    from agent.serve import build_default_graph  # lazy — cần deps retrieval + openai

    graph = build_default_graph()

    if args.chat:
        return _chat(graph, args.trace)
    if not args.question:
        parser.error("cần câu hỏi (hoặc dùng --chat)")

    answer = graph.ask(args.question)
    if args.json:
        print(json.dumps(answer.as_dict(), ensure_ascii=False, indent=2))
        return 0
    _print_answer(answer, show_trace=args.trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
