PYTHON ?= python

.PHONY: install sample data test lint up down index kg evals evals-live ask chat

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-retrieval:
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev,retrieval,llm]"

# infra local (Milvus + Neo4j) — cần Docker Desktop chạy
up:
	docker compose up -d

down:
	docker compose down

# 200 rows đầu từ HF (streaming) → data/processed/ — đủ để duyệt data
sample:
	$(PYTHON) -m pipelines.load_dataset --max-rows 200

# full dataset, tất cả split
data:
	$(PYTHON) -m pipelines.load_dataset

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

# M4: eval scenarios A–F (scripted, không cần key/infra)
evals:
	$(PYTHON) -m evals.run

# M4: eval với graph THẬT (cần LLM_API_KEY + Milvus/Neo4j) — evidence cho G1
evals-live:
	$(PYTHON) -m evals.run --live

# hỏi 1 câu end-to-end với components thật: make ask Q="bé sốt 3 ngày nên làm gì?"
ask:
	$(PYTHON) -m agent.cli "$(Q)"

# chat nhiều lượt (conversation) — câu follow-up hiểu ngữ cảnh câu trước
chat:
	$(PYTHON) -m agent.cli --chat

# M1: nạp vector index (subset 500 rows cho dev; bỏ --max-rows để chạy full)
index:
	$(PYTHON) -m pipelines.build_vector_index --max-rows 500

# M1: extract triplet -> Neo4j (mặc định 200 rows = 200 lượt gọi LLM)
kg:
	$(PYTHON) -m pipelines.build_kg --max-rows 200
