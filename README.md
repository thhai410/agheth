# vn-health-agent

Agent kiến trúc tham chiếu (reference architecture): **agentic RAG** — hybrid retrieval (Milvus vector + Neo4j knowledge graph), context assembly trên **DeepSeek Harness**, safety gate có self-correction loop, tool sandboxing. Demo domain: hỏi đáp y tế/sức khỏe tiếng Việt.

> **Demo / nghiên cứu — không phải sản phẩm y tế.** Output không thay thế tư vấn y khoa.
> Dataset demo là **CC BY-NC 4.0** — không thương mại hoá.

- Kiến trúc & quyết định đã chốt: [docs/architecture.md](docs/architecture.md)
- Trạng thái hiện tại: **M2+M3+M4 code xong** — full vector index (138,873 chunks) trong Milvus local;
  agent graph + safety gate + tools + eval scenarios A–F chạy được không cần API key (fakes).
  Còn: `make kg` + demo end-to-end cần API key (DeepSeek hoặc Gemini — xem `.env.example`).

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows (Git Bash)
pip install -e ".[dev]"
cp .env.example .env        # điền secrets sau — KHÔNG commit .env

make sample    # tải 200 rows mẫu từ HF → data/processed/ (cần mạng)
make data      # full dataset (~112k rows → train.parquet + val.parquet)
make test      # unit tests
make lint
```

### M1 — retrieval (cần Docker Desktop + deps ML)

```bash
make install-retrieval     # torch CPU + FlagEmbedding + pymilvus + neo4j (~1GB)
make up                    # docker compose: Milvus standalone + Neo4j
make index                 # chunk + BGE-M3 embed + nạp Milvus (subset 500 rows)
make kg                    # LLM extract triplet → Neo4j (cần LLM_API_KEY)
```

### Hỏi đáp end-to-end (cần LLM_API_KEY + infra)

```bash
make ask Q="bé sốt 3 ngày nên làm gì?"   # 1 câu qua full agent graph
make evals-live                          # eval scenarios với graph thật (evidence G1)
make kg                                  # build KG trước để có triplets khi retrieve
```

**Test trước bằng Gemini** (khi chưa có DeepSeek key): endpoint OpenAI-compatible của
Google — chỉ cần đổi 3 biến trong `.env` (xem `.env.example`), không cần sửa code:
`LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`,
`LLM_MODEL=gemini-3.8-flash`, `LLM_API_KEY=<key Google AI Studio>`.
Lưu ý model id Gemini hay đổi — kiểm tra danh sách model trên AI Studio.

## Bố cục

```
├── docs/architecture.md      # kiến trúc (sơ đồ theo ảnh leader, DSH, safety, roadmap)
├── src/agent/
│   ├── config.py             # settings từ .env (secrets không commit)
│   ├── llm.py                # LLM client OpenAI-compatible (DeepSeek/Gemini/...)
│   ├── data/clean.py         # hàm làm sạch data thuần (M0)
│   ├── data/chunk.py         # chunking sentence-aware cho vector index (M1)
│   ├── extract/              # intent & entities — DeepSeek V4.1        (M2)
│   ├── retrieval/            # Milvus hybrid (BGE-M3+BM25) + Neo4j KG   (M1)
│   ├── assemble/             # context assembly trên DSH                (M2)
│   ├── safety/               # gate + self-correction loop              (M3)
│   ├── tools/                # tool sandbox: drug_lookup, symptom_checker (M3)
│   └── serve.py              # FastAPI endpoint                         (M2)
├── pipelines/
│   ├── load_dataset.py       # HF → clean → Parquet + stats.json        (M0)
│   ├── build_vector_index.py # Parquet → chunk → BGE-M3 → Milvus        (M1)
│   └── build_kg.py           # Parquet → DeepSeek triplet → Neo4j       (M1)
├── evals/scenarios/          # kịch bản eval A–F                        (M4)
├── configs/data.yaml         # cấu hình dataset + cleaning rules
├── docker-compose.yml        # Milvus standalone + Neo4j                (M1)
└── tests/
```

## Secrets

`LLM_API_KEY` (hoặc `DEEPSEEK_API_KEY`) để trong `.env` (gitignored). Không commit,
không gửi key trong chat — nếu đã lỡ gửi thì xoá/rotate key.

## License note

Code: nội bộ. Dataset `NoirHuy/chatdoctor-healthcaremagic-112k-vi`: CC BY-NC 4.0 — chỉ dùng demo/nghiên cứu.
