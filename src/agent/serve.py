"""API endpoint — FastAPI (architecture.md §3).

    POST /v1/ask  { "question": "..." }
    -> { "answer": "...", "category": ..., "citations": [...], "trace_id": "..." }

Graph inject được — test truyền fake; production build từ Settings.
"""
from __future__ import annotations

from typing import Any


def build_default_graph():
    """Wire các implementation thật từ Settings — cần infra + LLM_API_KEY."""
    from agent.config import get_settings
    from agent.graph import AgentGraph
    from agent.llm import LlmClient
    from agent.retrieval.kg import Neo4jKgStore
    from agent.retrieval.vector import BgeM3Embedder, MilvusVectorStore
    from agent.tools.builtin import default_registry

    settings = get_settings()
    llm = LlmClient(
        settings.api_key, model=settings.model_id, base_url=settings.llm_base_url
    )
    embedder = BgeM3Embedder()
    retriever = MilvusVectorStore(uri=settings.milvus_uri, embedder=embedder)
    kg = Neo4jKgStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return AgentGraph.from_config(llm=llm, retriever=retriever, kg=kg, tools=default_registry())


def create_app(graph=None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(title="vn-health-agent (demo)", version="0.1.0")

    class AskRequest(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        # lịch sử hội thoại (tùy chọn): [{"role": "user"|"assistant", "content": "..."}]
        history: list[dict[str, str]] = Field(default_factory=list, max_length=40)

    class AskResponse(BaseModel):
        answer: str
        category: str
        citations: list[dict[str, Any]]
        trace_id: str
        attempts: int
        tool_results: dict[str, Any]
        trace: list[str]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:
        if graph is None:
            raise HTTPException(
                status_code=503, detail="chưa cấu hình graph (cần LLM_API_KEY + infra)"
            )
        return graph.ask(req.question, history=req.history).as_dict()

    return app


def main() -> None:  # pragma: no cover — chạy thật
    import uvicorn

    app = create_app(build_default_graph())
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
