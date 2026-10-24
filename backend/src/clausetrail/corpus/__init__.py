"""公开条款语料：列目录（不下载）、抽查链接、按需下载。

    clausetrail corpus list                 # 遍历四家官网目录，写 data/corpus/manifest.csv，只请求目录页
    clausetrail corpus check --per-source 5 # 每家抽几条链接，只读文件头确认是 PDF，不落盘
    clausetrail corpus download --limit 50  # 下载 50 份到 data/corpus/files/；不带参数时默认 20 份
    clausetrail corpus download --all       # 下载清单里的全部条款文件

全部下载约 6000 份 PDF、几个 GB，按每秒一个请求要几个小时。仓库不收录这些文件。
"""

import argparse
import csv
import errno
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import httpx

from clausetrail.config import PROJECT_ROOT
from clausetrail.corpus.fetch import MIN_DELAY, Blocked, Fetcher, NotPdf, free_gb, is_pdf_file
from clausetrail.corpus.sources import SOURCES, Clause, Skipped

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
FIELDS = ["key", "source", "product_id", "product", "status", "version", "url", "file"]
# 没加 --limit 也没加 --all 时只下载这么多，避免一条命令把磁盘写满
DEFAULT_LIMIT = 20
# 同一家连续失败这么多份就先停下这一家：多半是被限流或官网改版，继续请求只会给对方添负担
MAX_FAILURE_STREAK = 10
# 以这些字符开头的文本，表格软件打开清单时会当成公式执行
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _say(text: str) -> None:
    print(text, flush=True)


def _read(manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(manifest: Path, rows: list[dict]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(manifest)


def _cell(value) -> str:
    text = str(value)
    return "'" + text if text.startswith(FORMULA_PREFIXES) else text


def _row(c: Clause) -> dict:
    # 产品名、编号、版本来自网站，写进 CSV 前防公式注入；url 和 file 由本程序生成，不用处理
    return {
        "key": c.key,
        "source": c.source,
        "product_id": _cell(c.product_id),
        "product": _cell(c.product),
        "status": c.status,
        "version": _cell(c.version),
        "url": c.url,
        "file": c.file_name,
    }


def _referer(source: str) -> dict:
    s = SOURCES.get(source)
    return {"Referer": s.catalog} if s else {}


def _selected(names: list[str] | None) -> list[str]:
    # 去重：同一家写两遍时不遍历两遍，清单里也不会重复
    names = list(dict.fromkeys(names or SOURCES))
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"未知的来源：{', '.join(unknown)}；可选：{', '.join(SOURCES)}")
    return names


def cmd_list(args: argparse.Namespace, fetcher: Fetcher | None = None) -> int:
    names = _selected(args.sources)
    manifest = Path(args.manifest)
    previous = _read(manifest) if manifest.exists() else []
    f = fetcher or Fetcher(args.delay)
    collected: dict[str, list[dict]] = {}
    failed, partial = [], []
    try:
        for name in names:
            source = SOURCES[name]
            _say(f"[{name}] {source.label}：{source.catalog}")
            seen: set[str] = set()
            rows, skipped = [], []
            try:
                for item in source.walk(f, lambda msg, n=name: _say(f"  [{n}] {msg}")):
                    if isinstance(item, Skipped):
                        skipped.append(item)
                        _say(f"  [{name}] 跳过取不到的页面：{item.url}（{item.reason}）")
                    elif item.url not in seen:
                        seen.add(item.url)
                        rows.append(_row(item))
            except Exception as exc:
                # 官网改版时常见的是 KeyError、TypeError 这类，不只是网络错误；任何一家出错都不影响别家
                failed.append(name)
                _say(f"  [{name}] 失败：{type(exc).__name__}: {exc}。官网可能改版了，这一家保留上次的清单")
                continue
            if not rows:
                failed.append(name)
                _say(f"  [{name}] 一条条款也没有解析到，官网可能改版了，这一家保留上次的清单")
                continue
            if skipped:
                partial.append(name)
                # 这次一行都没取到的产品沿用上次清单里的行，免得清单缩水、之后的 download 漏下。
                # 同一产品编号在目录里可能有两行：只要有一行取到了，就以这次的结果为准，不把官网撤下的旧版本带回来
                missing = {_cell(s.product_id) for s in skipped if s.product_id} - {r["product_id"] for r in rows}
                restored = [
                    r for r in previous if r["source"] == name and r["product_id"] in missing and r["url"] not in seen
                ]
                rows += restored
                _say(
                    f"  [{name}] 有 {len(skipped)} 个页面取不到，已跳过，这些产品沿用上次清单里的 {len(restored)} 条；"
                    "重跑这一家可以补上"
                )
            before = sum(1 for r in previous if r["source"] == name)
            if len(rows) * 2 < before:
                _say(f"  [{name}] 这次 {len(rows)} 条，上次 {before} 条，少了一半以上，请确认官网是否改版")
            collected[name] = rows
    finally:
        if fetcher is None:
            f.close()

    # 只替换这次成功的来源，其余来源保留上次的结果：单独重跑失败的一家时不会丢掉别家的清单
    kept = [r for r in previous if r["source"] not in collected]
    rows = kept + [r for name in names for r in collected.get(name, [])]
    _write(manifest, rows)

    _say("")
    _say(f"{'来源':<16}{'产品':>8}{'条款文件':>10}")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for name, items in by_source.items():
        products = len({r["product_id"] or r["product"] for r in items})
        _say(f"{name:<16}{products:>8}{len(items):>10}")
    total_products = sum(len({r["product_id"] or r["product"] for r in v}) for v in by_source.values())
    _say(f"{'合计':<16}{total_products:>8}{len(rows):>10}")
    if failed:
        _say(f"没有更新的来源：{', '.join(failed)}（保留上次的清单）")
    _say(f"清单：{manifest}（只记录地址，没有下载文件）")
    return 1 if failed or partial else 0


def cmd_check(args: argparse.Namespace, fetcher: Fetcher | None = None) -> int:
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"找不到 {manifest}，先运行 clausetrail corpus list")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in _read(manifest):
        by_source[r["source"]].append(r)
    rng = random.Random(args.seed)
    f = fetcher or Fetcher(args.delay)
    bad = 0
    try:
        for name in _selected(args.sources):
            if not by_source.get(name):
                bad += 1
                _say(f"[{name}] 清单里没有这一家，先运行 clausetrail corpus list --sources {name}")
                continue
            sample = rng.sample(by_source[name], min(args.per_source, len(by_source[name])))
            ok = 0
            for r in sample:
                try:
                    status, ctype, head = f.head(r["url"], headers=_referer(name))
                    good = status == 200 and head.startswith(b"%PDF")
                    detail = f"{status} {ctype[:24]}"
                except (httpx.HTTPError, Blocked) as exc:
                    good, detail = False, f"{type(exc).__name__}: {exc}"
                ok += good
                if not good:
                    _say(f"  [{name}] 不可用：{detail} {r['url']}")
            bad += len(sample) - ok
            _say(f"[{name}] 抽查 {len(sample)} 条，{ok} 条返回 PDF")
    finally:
        if fetcher is None:
            f.close()
    return 1 if bad else 0


def cmd_download(args: argparse.Namespace, fetcher: Fetcher | None = None) -> int:
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"找不到 {manifest}，先运行 clausetrail corpus list")
    names = set(_selected(args.sources))
    listed = _read(manifest)
    rows = [r for r in listed if r["source"] in names]
    out = Path(args.out)
    limit = None if args.all else (args.limit if args.limit is not None else DEFAULT_LIMIT)
    f = fetcher or Fetcher(args.delay)
    done = skipped = 0
    failures: list[dict] = []
    streak: dict[str, int] = defaultdict(int)
    gave_up: set[str] = set()
    stopped = None
    try:
        for r in rows:
            source = r["source"]
            dest = out / r["file"]
            if is_pdf_file(dest):
                skipped += 1
                continue
            if source in gave_up:
                continue
            if limit is not None and done >= limit:
                break
            if free_gb(out) < args.min_free_gb:
                stopped = "disk"
                _say(f"磁盘剩余不足 {args.min_free_gb} GB，停止下载（已下载的文件保留，腾出空间后重跑会接着下）")
                break
            try:
                f.download(r["url"], dest, headers=_referer(source))
            except (httpx.HTTPError, Blocked, NotPdf, OSError) as exc:
                if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                    stopped = "disk"
                    _say("磁盘写满了，停止下载（已下载的文件保留，腾出空间后重跑会接着下）")
                    break
                failures.append({**r, "error": f"{type(exc).__name__}: {exc}"})
                _say(f"  失败：{source} {r['product']}：{type(exc).__name__}")
                streak[source] += 1
                if streak[source] >= MAX_FAILURE_STREAK:
                    gave_up.add(source)
                    _say(
                        f"  [{source}] 连续 {MAX_FAILURE_STREAK} 份失败，这一家先停下：可能被限流或官网改版了，稍后再试"
                    )
            else:
                streak[source] = 0
                done += 1
                if done % 50 == 0:
                    _say(f"  已下载 {done} 份")
    except KeyboardInterrupt:
        stopped = "interrupt"
        _say("已中断，已下载的文件保留，重跑会跳过它们")
    finally:
        if fetcher is None:
            f.close()

    # failures.csv 只记最近一次运行的失败；这次没有失败就删掉上次的，免得留着过期的记录
    failures_csv = out / "failures.csv"
    if failures:
        out.mkdir(parents=True, exist_ok=True)
        with failures_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[*FIELDS, "error"], extrasaction="ignore")
            w.writeheader()
            w.writerows(failures)
        _say(f"失败的 {len(failures)} 条记在 {failures_csv}")
    else:
        failures_csv.unlink(missing_ok=True)
    _say(f"下载 {done} 份，已存在跳过 {skipped} 份，失败 {len(failures)} 份；文件在 {out}")
    remaining = sum(1 for r in rows if not is_pdf_file(out / r["file"]))
    if remaining:
        if stopped:
            hint = "重跑会接着下"
        elif limit is not None and done >= limit:
            hint = "调大 --limit 或用 --all 继续"
        else:
            hint = "重跑会再试失败的文件"
        _say(f"还有 {remaining} 份没下载：{hint}")
    if out.is_dir():
        files = {r["file"] for r in listed}
        orphans = [p for p in out.glob("*.pdf") if p.name not in files]
        if orphans:
            _say(f"{out} 里有 {len(orphans)} 份 PDF 不在当前清单里（官网下架或换了地址），入库前可以先删掉")
    if done or skipped:
        _say(f"入库：clausetrail ingest {out}")
    if stopped == "interrupt":
        return 130
    return 1 if failures or stopped or gave_up else 0


def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"需要正整数，收到的是 {text}")
    return n


def _finite(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = math.nan
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"需要有限的数，收到的是 {text}")
    return v


def add_parser(sub) -> None:
    p = sub.add_parser("corpus", help="公开条款语料：列目录、抽查链接、下载")
    acts = p.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sources", nargs="+", metavar="SOURCE", help=f"只处理这几家：{' '.join(SOURCES)}")
    common.add_argument("--manifest", default=str(CORPUS_DIR / "manifest.csv"))
    common.add_argument(
        "--delay", type=_finite, default=1.0, help=f"同一主机两次请求的最小间隔（秒，不低于 {MIN_DELAY:g}）"
    )
    acts.add_parser("list", parents=[common], help="遍历官网目录，写出条款清单（不下载文件）")
    c = acts.add_parser("check", parents=[common], help="每家抽查几条链接，确认返回的是 PDF（不落盘）")
    c.add_argument("--per-source", type=_positive_int, default=5)
    c.add_argument("--seed", type=int, default=0)
    d = acts.add_parser("download", parents=[common], help=f"下载条款 PDF（默认只下 {DEFAULT_LIMIT} 份）")
    d.add_argument("--out", default=str(CORPUS_DIR / "files"))
    how_many = d.add_mutually_exclusive_group()
    how_many.add_argument("--limit", type=_positive_int, help="这次最多下载几份（已下载的不算）")
    how_many.add_argument("--all", action="store_true", help="下载清单里的全部文件")
    d.add_argument("--min-free-gb", type=_finite, default=2.0, help="磁盘剩余低于这个值就停止")


def run(args: argparse.Namespace) -> int:
    if args.delay < MIN_DELAY:
        _say(f"--delay 不能低于 {MIN_DELAY:g} 秒，已按 {MIN_DELAY:g} 秒执行")
        args.delay = MIN_DELAY
    started = time.monotonic()
    code = {"list": cmd_list, "check": cmd_check, "download": cmd_download}[args.action](args)
    _say(f"用时 {time.monotonic() - started:.0f} 秒")
    return code
