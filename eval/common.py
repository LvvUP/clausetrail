import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from clausetrail.config import get_settings
from clausetrail.services import Services, ServiceUnavailable, build_services

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "eval" / "datasets"
REPORTS = ROOT / "eval" / "reports"
SAMPLES = ROOT / "data" / "samples"
OFFLINE_HINT = "离线演示可以设置 EMBEDDER_BACKEND=fake RERANKER_BACKEND=fake LLM_BACKEND=fake"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def backends(services: Services) -> dict[str, str]:
    return {
        "embedder": getattr(services.embedder, "name", "-"),
        "reranker": getattr(services.reranker, "name", "-"),
        "llm": getattr(services.llm, "name", "-"),
        "bm25": services.bm25.name,
    }


def header(title: str, docs: int, queries: int | None, backend: dict[str, str]) -> str:
    scale = f"{docs} 份文档" + (f" / {queries} 条 query" if queries is not None else "")
    lines = [
        f"# {title}",
        "",
        f"- 样例规模：{scale}",
        "- 模型后端：" + "，".join(f"{k}={v}" for k, v in backend.items()),
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
    ]
    if any(v.startswith("fake") for v in backend.values()):
        lines.append("- **使用离线示例模型（fake 后端），指标仅作示例**")
    return "\n".join(lines) + "\n"


def gold_docs(rows: list[dict]) -> set[str]:
    return {g["doc_id"] for r in rows for g in r.get("gold", [])}


def ensure_indexed(services: Services, rows: list[dict]) -> None:
    indexed = {d["doc_id"] for d in services.db.list_documents()}
    missing = gold_docs(rows) - indexed
    if missing:
        raise SystemExit(f"索引里缺少 {len(missing)} 份评测文档，先运行 make ingest")


def run_with_services(fn: Callable[[Services], Awaitable[object]], *components: str) -> int:
    services = build_services(get_settings())

    # 异步的模型客户端要在 asyncio.run 的事件循环结束前关闭
    async def main() -> None:
        try:
            await fn(services)
        finally:
            await services.aclose()

    try:
        services.require(*components)
        asyncio.run(main())
        return 0
    except ServiceUnavailable as exc:
        print(f"[评测中止] {exc}\n{OFFLINE_HINT}")
        return 2
    finally:
        services.close()
