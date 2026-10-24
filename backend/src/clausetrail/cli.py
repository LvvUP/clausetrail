import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from clausetrail.config import get_settings


def _files(paths: list[str]) -> list[Path]:
    from clausetrail.ingest.pipeline import SUPPORTED_SUFFIXES

    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_SUFFIXES))
        elif p.is_file():
            out.append(p)
        else:
            print(f"[skip] 找不到 {p}", file=sys.stderr)
    return out


async def _ingest(services, args) -> int:
    from clausetrail.ingest.pipeline import ingest_file

    failed = 0
    for path in _files(args.paths):
        try:
            report = await ingest_file(services, path)
            print(
                f"[ok] {path.name}: {report.pages} 页, {report.chunks} 个片段, 解析器 {report.parsers}, "
                f"OCR 纠错 {report.ocr_fixes} 处, kb_version={report.kb_version}, {report.seconds}s"
            )
        except Exception as exc:
            failed += 1
            print(f"[fail] {path.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


async def _remove(services, args) -> int:
    from clausetrail.ingest.pipeline import remove_document

    try:
        ok = await remove_document(services, args.doc_id)
    except ValueError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"[fail] 删除没有完成（{type(exc).__name__}: {exc}）。文档记录还在，排除问题后重新执行 remove 即可",
            file=sys.stderr,
        )
        return 1
    print("[ok] 已删除" if ok else "[skip] 文档列表里没有这个 doc_id（索引里如有残留已一并清掉）")
    return 0 if ok else 1


async def _docs(services, args) -> int:
    print(f"kb_version={services.db.kb_version()}")
    for d in services.db.list_documents():
        print(f"{d['doc_id']}  {d['pages']:>3} 页  {d['chunks']:>4} 片段  {d['file_name']}")
    return 0


async def _search(services, args) -> int:
    from clausetrail.retrieval.pipeline import Retriever

    result = await Retriever(services).retrieve(args.query)
    print(f"dense={result.dense_hits} bm25={result.bm25_hits}({services.bm25.name}) timings={result.timings_ms}")
    for i, c in enumerate(result.candidates[: args.k], start=1):
        ranks = f"dense#{c.dense_rank or '-'} bm25#{c.bm25_rank or '-'}"
        print(
            f"{i:>2}. rerank={c.rerank_score:.3f} rrf={c.rrf_score:.4f} {ranks}  "
            f"{c.chunk.file_name} · {c.chunk.section_label} · p{c.chunk.page_start}"
        )
        print(f"    {c.chunk.text[:120].replace(chr(10), ' ')}")
    return 0


async def _ask(services, args) -> int:
    from clausetrail.chat import ChatService

    chat = ChatService(services)
    citations: list[dict] = []
    code = 0
    async for name, data in chat.stream(args.question, args.session):
        if name == "meta":
            if data["rewritten_query"]:
                print(f"（已理解为：{data['rewritten_query']}）")
            if data["cache_hit"]:
                print("（命中缓存）")
        elif name == "retrieval":
            citations = data["citations"]
        elif name == "delta":
            print(data["text"], end="", flush=True)
        elif name == "refusal":
            print(f"\n[拒答] {data['message']}")
        elif name == "error":
            print(f"\n[错误] {data['message']}", file=sys.stderr)
            code = 1
        elif name == "done":
            print()
            for c in citations:
                if c["n"] in data["used_citations"]:
                    print(f"  [{c['n']}] {c['file']} · {' > '.join(c['section_path'])} · 第{c['page_start']}页")
            print(f"session={data['session_id']} timings={data['timings_ms']}")
    await chat.drain()
    return code


async def _health(services, args) -> int:
    from clausetrail.api.routes.health import health_report

    report = health_report(services)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


COMMANDS = {"ingest": _ingest, "remove": _remove, "docs": _docs, "search": _search, "ask": _ask, "health": _health}


async def _dispatch(args) -> int:
    from clausetrail.services import build_services

    services = build_services()
    try:
        return await COMMANDS[args.command](services, args)
    finally:
        await services.aclose()
        services.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clausetrail", description="款知 ClauseTrail：保险条款问答")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="解析条款并写入索引")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("remove", help="从索引中删除文档")
    p.add_argument("doc_id")
    sub.add_parser("docs", help="列出已入库文档")
    p = sub.add_parser("search", help="查看检索结果（不调用生成模型）")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("ask", help="在终端里提问")
    p.add_argument("question")
    p.add_argument("--session")
    sub.add_parser("health", help="检查各组件状态")
    p = sub.add_parser("serve", help="启动 API 服务")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    from clausetrail.corpus import add_parser as add_corpus_parser

    add_corpus_parser(sub)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 语料命令只发 HTTP 请求，不需要模型和索引，不走 build_services
    if args.command == "corpus":
        from clausetrail.corpus import run as run_corpus

        return run_corpus(args)
    if args.command == "serve":
        import uvicorn

        settings = get_settings()
        uvicorn.run(
            "clausetrail.api.app:run",
            factory=True,
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            reload=args.reload,
        )
        return 0
    return asyncio.run(_dispatch(args))


if __name__ == "__main__":
    sys.exit(main())
