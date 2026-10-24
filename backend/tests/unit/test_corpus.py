import argparse
import csv
import errno
import json

import httpx
import pytest

from clausetrail import cli
from clausetrail.corpus import FIELDS, _row, _selected, cmd_check, cmd_download, cmd_list
from clausetrail.corpus import fetch as fetch_module
from clausetrail.corpus.fetch import ROBOTS_RETRY_AFTER, Blocked, Fetcher, NotPdf
from clausetrail.corpus.sources import (
    CHINALIFE_LIST,
    CPIC_ROOT,
    NCL_LIST,
    PINGAN_LIST,
    Clause,
    chinalife_life,
    cpic_life,
    newchinalife,
    pingan_life,
)

PDF = b"%PDF-1.7\n" + b"x" * 100


def fetcher(handler) -> Fetcher:
    def route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return handler(req)

    return Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))


def walk(adapter, handler) -> list:
    return list(adapter(fetcher(handler), lambda msg: None))


def test_pingan_lists_every_version_with_a_clause():
    def handler(req):
        assert str(req.url) == PINGAN_LIST and json.loads(req.content)["isOnlyNew"] == "N"
        if json.loads(req.content)["isOrNotSale"] == "Y":
            data = [{"planCode": "866", "actualPlanCode": "866/866D", "versionNo": "866-2", "clauseContent": "1"}]
        else:
            data = [
                {"planCode": "1031", "versionNo": "1031-1", "clauseContent": "0", "endDate": "2015-01-01"},
                {"planCode": "2057", "versionNo": "2057-2", "clauseContent": "0", "endDate": "2020-01-01"},
            ]
        return httpx.Response(200, json={"CODE": "00", "DATA": data})

    got = walk(pingan_life, handler)
    # 2017-04-01 之后、clauseContent 不是 1 的没有条款
    assert [(c.status, c.version) for c in got] == [("在售", "866-2"), ("停售", "1031-1")]
    assert got[0].url.endswith("planCode=866/866D&versionNo=866-2&attachmentType=1")


def test_chinalife_pages_until_count_and_skips_missing_clauses():
    def handler(req):
        assert str(req.url).startswith(CHINALIFE_LIST)
        page, kind = int(req.url.params["page"]), req.url.params["type"]
        if kind == "2":
            return httpx.Response(200, json={"count": 0, "list": []})
        items = [{"id": f"p{page}", "productname": f"产品{page}", "clause": f"/upload/{page}.pdf"}]
        if page == 2:
            items.append({"id": "x", "productname": "无条款", "clause": "/"})
        return httpx.Response(200, json={"count": 3, "list": items})

    got = walk(chinalife_life, handler)
    assert [c.url for c in got] == [
        "https://www.e-chinalife.com/upload/1.pdf",
        "https://www.e-chinalife.com/upload/2.pdf",
    ]


NCL_PAGE = """
<table><tr><td>序号</td><td>产品名称</td><td>险种</td><td>状态</td><td>材料</td></tr>
<tr><td>1</td><td>新华甲</td><td>健康险</td><td>在售</td><td>
  <a href="/node/670?riskCode=A1&amp;status=1" class="dropdown-item"> 条款 </a>
  <a href="https://static-cdn.newchinalife.com/ncl/pdf/rate.pdf" class="dropdown-item"> 费率表 </a></td></tr>
<tr><td>2</td><td>新华乙</td><td>寿险</td><td>停售</td><td>
  <a href="https://static-cdn.newchinalife.com/ncl/pdf/b.pdf" class="dropdown-item"> 条款 </a></td></tr>
</table><span class="sum">共{pages}页，3条</span>
"""
NCL_DETAIL = """
<div class=" insuranceTable tlInsurance"><table><thead><tr><td>序号</td><td>条款名称</td><td>停用时间</td></tr></thead>
<tbody><tr><td>1</td><td><a href="https://static-cdn.newchinalife.com/ncl/pdf/a2.pdf">新华甲</a></td><td></td></tr>
<tr><td>2</td><td><a href="https://static-cdn.newchinalife.com/ncl/pdf/a1.pdf">新华甲</a></td><td>2024-03-01</td></tr>
</tbody></table></div>
"""


def test_newchinalife_reads_every_clause_version_from_detail_pages():
    def handler(req):
        if req.url.path == "/node/372":
            return httpx.Response(200, text=NCL_PAGE.replace("{pages}", "2"))
        if req.url.path == "/node/372_2":
            return httpx.Response(200, text=NCL_PAGE.replace("{pages}", "2").replace("新华乙", "新华丙"))
        if req.url.path == "/node/670":
            return httpx.Response(200, text=NCL_DETAIL)
        raise AssertionError(req.url)

    got = walk(newchinalife, handler)
    first = [(c.product, c.version, c.url.rsplit("/", 1)[-1], c.product_id) for c in got[:3]]
    assert first == [
        ("新华甲", "当前", "a2.pdf", "A1"),
        ("新华甲", "停用于 2024-03-01", "a1.pdf", "A1"),
        ("新华乙", "当前", "b.pdf", "新华乙"),
    ]
    assert len(got) == 6 and NCL_LIST.endswith("/node/372")


CPIC_PAGE = """
<script>if(ele.value>{pages}){{}}</script>
<table id="example"><thead><tr><th>序号</th><th>产品名称及条款</th></tr></thead><tbody>
<tr><td>1</td><td>
  <a href="/mpdfjs/pdf.html?file=/upload/resources/file/{n}.pdf" title="太保{n}" target="_blank">太保{n}</a>
</td></tr>
</tbody></table>
"""


def test_cpic_walks_every_category_and_page_and_unwraps_the_viewer_link():
    def handler(req):
        path = req.url.path
        if path.endswith("/zbcp/jkx/"):
            return httpx.Response(200, text=CPIC_PAGE.format(pages=2, n="a"))
        if path.endswith("/zbcp/jkx/index_2.shtml"):
            return httpx.Response(200, text=CPIC_PAGE.format(pages=2, n="b"))
        assert path.startswith("/xrsbx/") and str(req.url).startswith(CPIC_ROOT)
        return httpx.Response(200, text='<table id="example"></table>')

    got = walk(cpic_life, handler)
    assert [(c.product, c.status, c.url) for c in got] == [
        ("太保a", "在售", "https://life.cpic.com.cn/upload/resources/file/a.pdf"),
        ("太保b", "在售", "https://life.cpic.com.cn/upload/resources/file/b.pdf"),
    ]


def test_robots_rules_are_respected():
    def route(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(
                200, text="User-agent: *\nDisallow: /private/", headers={"content-type": "text/plain"}
            )
        return httpx.Response(200, content=PDF)

    f = Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))
    assert f.allowed("https://a.example/upload/x.pdf")
    with pytest.raises(Blocked):
        f.get("https://a.example/private/x.pdf")


def test_download_keeps_only_pdfs_and_skips_existing(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, content=b"<html>login</html>" if "bad" in req.url.path else PDF)

    f = fetcher(handler)
    dest = tmp_path / "a.pdf"
    assert f.download("https://a.example/ok.pdf", dest) == len(PDF)
    assert f.download("https://a.example/ok.pdf", dest) == 0
    with pytest.raises(NotPdf):
        f.download("https://a.example/bad.pdf", tmp_path / "b.pdf")
    assert not (tmp_path / "b.pdf").exists() and not (tmp_path / "b.pdf.part").exists()
    assert calls == ["/ok.pdf", "/bad.pdf"]


def test_download_command_stops_at_limit_and_resumes(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i in range(3):
            w.writerow(
                {"key": f"k{i}", "source": "cpic-life", "url": f"https://a.example/{i}.pdf", "file": f"k{i}.pdf"}
            )
    out = tmp_path / "files"

    def args(**kw):
        base = dict(manifest=str(manifest), out=str(out), sources=None, limit=2, all=False, min_free_gb=0, delay=0)
        return argparse.Namespace(**{**base, **kw})

    f = fetcher(lambda req: httpx.Response(200, content=PDF))
    assert cmd_download(args(), f) == 0
    assert sorted(p.name for p in out.iterdir()) == ["k0.pdf", "k1.pdf"]
    assert cmd_download(args(), f) == 0
    assert sorted(p.name for p in out.iterdir()) == ["k0.pdf", "k1.pdf", "k2.pdf"]


def test_list_writes_manifest_and_keeps_other_sources(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({"key": "old", "source": "cpic-life", "url": "https://a.example/old.pdf", "file": "old.pdf"})

    def handler(req):
        if req.url.params.get("type") == "2":
            return httpx.Response(200, json={"count": 0, "list": []})
        return httpx.Response(
            200, json={"count": 1, "list": [{"id": "1", "productname": "国寿甲", "clause": "/a.pdf"}]}
        )

    ns = argparse.Namespace(sources=["chinalife-life"], manifest=str(manifest), delay=0)
    assert cmd_list(ns, fetcher(handler)) == 0
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["source"] for r in rows] == ["cpic-life", "chinalife-life"]


class FakeClock:
    """替换 fetch 模块里的 time：退避和限速不真的等待，测试可以查看等了多久。"""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(fetch_module, "time", c)
    return c


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download_args(manifest, out, **kw):
    base = dict(manifest=str(manifest), out=str(out), sources=None, limit=None, all=True, min_free_gb=0, delay=0)
    return argparse.Namespace(**{**base, **kw})


def test_robots_failure_blocks_the_site_for_a_while_then_rechecks(clock):
    robots = {"status": 503}
    hits = []

    def route(req):
        hits.append(req.url.path)
        if req.url.path == "/robots.txt":
            return httpx.Response(robots["status"], text="User-agent: *\nDisallow: /private/")
        return httpx.Response(200, content=PDF)

    f = Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))
    with pytest.raises(Blocked):
        f.get("https://a.example/x.pdf")
    # 5xx 重试 3 次后按全部不允许；冷却期内不再请求这个站点
    with pytest.raises(Blocked):
        f.get("https://a.example/x.pdf")
    assert hits == ["/robots.txt"] * 3
    clock.now += ROBOTS_RETRY_AFTER
    robots["status"] = 200
    assert f.get("https://a.example/x.pdf").status_code == 200
    assert not f.allowed("https://a.example/private/y.pdf")


def test_robots_rules_apply_whatever_the_content_type(clock):
    def route(req):
        if req.url.path == "/robots.txt" and req.url.host == "a.example":
            body = b"User-agent: *\nDisallow: /private/"
            return httpx.Response(200, content=body, headers={"content-type": "application/octet-stream"})
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="<html>页面不存在</html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=PDF)

    f = Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))
    assert not f.allowed("https://a.example/private/x.pdf")
    assert f.allowed("https://a.example/x.pdf") and f.allowed("https://b.example/private/x.pdf")


def test_redirects_are_followed_hop_by_hop_under_robots_rules(tmp_path, clock):
    hits = []

    def route(req):
        hits.append(f"{req.url.scheme}://{req.url.host}{req.url.path}")
        if req.url.path == "/robots.txt":
            return httpx.Response(
                200, text="User-agent: *\nDisallow: /secret/" if req.url.host == "cdn.example" else ""
            )
        targets = {
            "/moved.pdf": "https://cdn.example/files/x.pdf",
            "/hidden.pdf": "https://cdn.example/secret/x.pdf",
            "/insecure.pdf": "http://a.example/x.pdf",
        }
        if req.url.path in targets:
            return httpx.Response(302, headers={"Location": targets[req.url.path]})
        return httpx.Response(200, content=PDF)

    f = Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))
    assert f.download("https://a.example/moved.pdf", tmp_path / "a.pdf") == len(PDF)
    assert "https://cdn.example/robots.txt" in hits
    with pytest.raises(Blocked):
        f.download("https://a.example/hidden.pdf", tmp_path / "b.pdf")
    with pytest.raises(Blocked):
        f.get("https://a.example/insecure.pdf")
    assert "https://cdn.example/secret/x.pdf" not in hits and "http://a.example/x.pdf" not in hits


def test_retries_honour_retry_after_and_do_not_sleep_after_the_last_attempt(clock):
    answers = [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text="ok")]
    assert fetcher(lambda req: answers.pop(0)).get("https://a.example/x").text == "ok"
    assert clock.sleeps == [7.0]
    clock.sleeps.clear()
    with pytest.raises(httpx.HTTPError):
        fetcher(lambda req: httpx.Response(503)).get("https://a.example/y")
    assert clock.sleeps == [1.0, 2.0]


def test_list_keeps_previous_rows_when_a_source_breaks_or_comes_back_empty(tmp_path, clock):
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {"key": "old-pa", "source": "pingan-life", "url": "https://a.example/1.pdf", "file": "1.pdf"},
            {"key": "old-cpic", "source": "cpic-life", "url": "https://a.example/2.pdf", "file": "2.pdf"},
        ],
    )

    def handler(req):
        if req.url.path.endswith("getProductList"):
            # 改版后最常见的是字段缺失：这里没有 planCode，会抛 KeyError
            return httpx.Response(200, json={"CODE": "00", "DATA": [{"clauseContent": "1"}]})
        return httpx.Response(200, text='<table id="example"></table>')

    ns = argparse.Namespace(sources=["pingan-life", "cpic-life"], manifest=str(manifest), delay=0)
    assert cmd_list(ns, fetcher(handler)) == 1
    assert [r["key"] for r in read_csv(manifest)] == ["old-pa", "old-cpic"]


def test_cpic_reports_a_changed_page_instead_of_returning_nothing():
    with pytest.raises(RuntimeError):
        walk(cpic_life, lambda req: httpx.Response(200, text="<html>改版后的页面</html>"))


def test_newchinalife_skips_detail_pages_it_cannot_fetch(tmp_path, clock):
    def handler(req):
        if req.url.path == "/node/372":
            return httpx.Response(200, text=NCL_PAGE.replace("{pages}", "1"))
        return httpx.Response(404)

    manifest = tmp_path / "manifest.csv"
    old = {"source": "newchinalife", "product_id": "A1", "product": "新华甲", "file": "a.pdf"}
    write_manifest(
        manifest,
        [
            {**old, "key": "a2", "url": "https://static-cdn.newchinalife.com/ncl/pdf/a2.pdf"},
            {**old, "key": "a1", "url": "https://static-cdn.newchinalife.com/ncl/pdf/a1.pdf"},
            {**old, "key": "gone", "product_id": "Z9", "url": "https://static-cdn.newchinalife.com/ncl/pdf/z.pdf"},
        ],
    )
    ns = argparse.Namespace(sources=["newchinalife"], manifest=str(manifest), delay=0)
    assert cmd_list(ns, fetcher(handler)) == 1
    # 取不到详情页的新华甲沿用上次的两条；已经不在目录里的产品不保留
    rows = read_csv(manifest)
    assert [r["product"] for r in rows] == ["新华乙", "新华甲", "新华甲"]
    assert [r["key"] for r in rows[1:]] == ["a2", "a1"]


def test_restored_rows_skip_products_that_were_partly_fetched(tmp_path, clock):
    # 同一产品编号 A1 在目录里有两行：一行的详情页取到了，另一行取不到
    page = NCL_PAGE.replace("{pages}", "1").replace(
        '<a href="https://static-cdn.newchinalife.com/ncl/pdf/b.pdf" class="dropdown-item"> 条款 </a>',
        '<a href="/node/671?riskCode=A1&amp;status=2" class="dropdown-item"> 条款 </a>',
    )

    def handler(req):
        if req.url.path == "/node/372":
            return httpx.Response(200, text=page)
        if req.url.path == "/node/670":
            return httpx.Response(200, text=NCL_DETAIL)
        return httpx.Response(404)

    manifest = tmp_path / "manifest.csv"
    withdrawn = "https://static-cdn.newchinalife.com/ncl/pdf/withdrawn.pdf"
    write_manifest(
        manifest,
        [{"key": "w", "source": "newchinalife", "product_id": "A1", "url": withdrawn, "file": "w.pdf"}],
    )
    ns = argparse.Namespace(sources=["newchinalife"], manifest=str(manifest), delay=0)
    assert cmd_list(ns, fetcher(handler)) == 1
    urls = [r["url"].rsplit("/", 1)[-1] for r in read_csv(manifest)]
    assert urls == ["a2.pdf", "a1.pdf"]


def test_list_reports_pages_that_change_midway(clock):
    def ncl(req):
        if req.url.path == "/node/372":
            return httpx.Response(200, text=NCL_PAGE.replace("{pages}", "2"))
        if req.url.path == "/node/670":
            return httpx.Response(200, text=NCL_DETAIL)
        return httpx.Response(200, text="<html>改版后的第 2 页</html>")

    with pytest.raises(RuntimeError):
        walk(newchinalife, ncl)

    def chinalife(req):
        page = int(req.url.params["page"])
        items = [{"id": "1", "productname": "国寿甲", "clause": "/1.pdf"}] if page == 1 else []
        return httpx.Response(200, json={"count": 5, "list": items})

    with pytest.raises(RuntimeError):
        walk(chinalife_life, chinalife)


def test_pingan_accepts_numeric_codes():
    def handler(req):
        data = [{"planCode": 866, "versionNo": 2, "clauseContent": "1", "clauseName": "平安甲"}]
        return httpx.Response(
            200, json={"CODE": "00", "DATA": data if json.loads(req.content)["isOrNotSale"] == "Y" else []}
        )

    (got,) = walk(pingan_life, handler)
    assert got.url.endswith("planCode=866&versionNo=2&attachmentType=1")
    assert _row(got)["product_id"] == "866" and got.file_name.endswith("-平安甲.pdf")


def test_robots_redirects_are_throttled_and_never_downgraded(clock):
    hits = []

    def route(req):
        hits.append(f"{req.url.scheme}://{req.url.host}{req.url.path}")
        if req.url.host == "a.example" and req.url.path == "/robots.txt":
            return httpx.Response(301, headers={"Location": "http://a.example/robots.txt"})
        if req.url.host == "b.example" and req.url.path == "/robots.txt":
            return httpx.Response(301, headers={"Location": "https://c.example/robots-b.txt"})
        if req.url.path == "/robots-b.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
        return httpx.Response(200, content=PDF)

    f = Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(route)))
    # 降级到 http 不跟随：按取不到 robots.txt 处理，不崩溃
    assert not f.allowed("https://a.example/x.pdf")
    assert "http://a.example/robots.txt" not in hits
    # 跳到别的主机取到的规则，仍然用于原站点
    assert not f.allowed("https://b.example/private/x.pdf") and f.allowed("https://b.example/x.pdf")
    assert "https://c.example/robots-b.txt" in hits


def test_newchinalife_detail_links_may_be_relative_or_uppercase():
    detail = NCL_DETAIL.replace("https://static-cdn.newchinalife.com/ncl/pdf/a1.pdf", "/ncl/pdf/a1.PDF?v=2")

    def handler(req):
        if req.url.path == "/node/372":
            return httpx.Response(200, text=NCL_PAGE.replace("{pages}", "1"))
        return httpx.Response(200, text=detail)

    assert "https://www.newchinalife.com/ncl/pdf/a1.PDF?v=2" in [c.url for c in walk(newchinalife, handler)]


def test_download_pauses_a_failing_site_and_keeps_only_fresh_failures(tmp_path, clock):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"key": f"c{i}", "source": "cpic-life", "url": f"https://bad.example/{i}.pdf", "file": f"c{i}.pdf"}
        for i in range(12)
    ]
    rows.append({"key": "p0", "source": "pingan-life", "url": "https://good.example/0.pdf", "file": "p0.pdf"})
    write_manifest(manifest, rows)
    out = tmp_path / "files"
    hits = []

    def handler(req):
        hits.append(req.url.host)
        return httpx.Response(403) if req.url.host == "bad.example" else httpx.Response(200, content=PDF)

    assert cmd_download(download_args(manifest, out), fetcher(handler)) == 1
    # 连续 10 份失败后这一家停下，别家照常下载
    assert hits.count("bad.example") == 10 and (out / "p0.pdf").exists()
    assert len(read_csv(out / "failures.csv")) == 10
    assert cmd_download(download_args(manifest, out), fetcher(lambda req: httpx.Response(200, content=PDF))) == 0
    assert not (out / "failures.csv").exists()


def test_download_records_file_system_errors_and_stops_when_the_disk_is_full(tmp_path, monkeypatch, clock):
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {"key": f"k{i}", "source": "cpic-life", "url": f"https://a.example/{i}.pdf", "file": f"k{i}.pdf"}
            for i in range(3)
        ],
    )
    out = tmp_path / "files"
    f = fetcher(lambda req: httpx.Response(200, content=PDF))
    tried = []

    def flaky(url, dest, headers=None):
        tried.append(url[-5:])
        raise PermissionError(errno.EACCES, "denied") if url.endswith("0.pdf") else OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(f, "download", flaky)
    assert cmd_download(download_args(manifest, out), f) == 1
    assert tried == ["0.pdf", "1.pdf"]
    assert [r["key"] for r in read_csv(out / "failures.csv")] == ["k0"]

    def interrupted(url, dest, headers=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(f, "download", interrupted)
    assert cmd_download(download_args(manifest, out), f) == 130


def test_check_flags_sources_missing_from_the_manifest(tmp_path, clock):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [{"key": "c", "source": "cpic-life", "url": "https://a.example/c.pdf", "file": "c.pdf"}])
    hits = []
    f = fetcher(lambda req: hits.append(req) or httpx.Response(200, content=PDF))
    ns = argparse.Namespace(manifest=str(manifest), sources=["pingan-life"], per_source=5, seed=0, delay=0)
    assert cmd_check(ns, f) == 1 and hits == []


@pytest.mark.parametrize(
    "argv",
    [
        ["download", "--limit", "0"],
        ["download", "--limit", "-1"],
        ["download", "--all", "--limit", "5"],
        ["list", "--delay", "nan"],
        ["check", "--per-source", "0"],
        ["download", "--min-free-gb", "nan"],
    ],
)
def test_corpus_command_line_rejects_bad_values(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(["corpus", *argv])
    assert exc.value.code == 2


def test_file_names_cannot_escape_the_folder_and_csv_cells_are_not_formulas():
    c = Clause("cpic-life", "../太保 安心(A)/=1+1", "在售", "当前", "https://a.example/x.pdf", product_id="=cmd")
    assert c.file_name == f"{c.key}-太保_安心(A)_1_1.pdf"
    nameless = Clause("cpic-life", "-", "在售", "当前", "https://a.example/y.pdf")
    assert nameless.file_name == f"{nameless.key}.pdf"
    row = _row(c)
    assert row["product_id"] == "'=cmd" and row["product"] == c.product
    assert _selected(["cpic-life", "cpic-life"]) == ["cpic-life"]
