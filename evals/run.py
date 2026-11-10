"""CLI: python -m evals.run [--live] — chạy tất cả scenarios A–F.

--live: graph THẬT (cần LLM_API_KEY + infra) — evidence cho G1.
"""
from evals.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
