"""四家寿险公司官网「公开信息披露 → 产品基本信息」的条款目录。

每个函数遍历一家公司的目录，产出条款 PDF 的地址，不下载文件。接口和页面结构是 2026-10-22 核实的，
官网改版后对应的函数可能失效：找不到预期的接口字段或页面结构时直接报错，`clausetrail corpus list` 会报出是哪一家。
历史版本的条款单独算一份：同一产品修订过条款时，各版本都是单独备案、单独公开的文件。
"""

import hashlib
import html
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import httpx

from clausetrail.corpus.fetch import Blocked, Fetcher

ON_SALE, STOPPED = "在售", "停售"


@dataclass
class Clause:
    source: str
    product: str
    status: str
    version: str
    url: str
    product_id: str = ""
    # 有的站点要带 Referer 才返回文件
    headers: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}-{hashlib.sha1(self.url.encode()).hexdigest()[:16]}"

    @property
    def file_name(self) -> str:
        # 带上产品名，入库后引用卡片上能看出是哪份条款；只保留文字、数字和括号，网站内容决定不了路径
        name = re.sub(r"[^\w（）()-]+", "_", self.product).strip("_-")[:40]
        return f"{self.key}-{name}.pdf" if name else f"{self.key}.pdf"


@dataclass
class Skipped:
    """某个产品页取不到：跳过它，同一家的其他产品照常写入清单。"""

    url: str
    reason: str
    # 被跳过的产品：list 用它从上次的清单里找回这个产品的行
    product_id: str = ""


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _cells(row: str) -> list[str]:
    return re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)


def _rows(table: str) -> list[str]:
    return re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)


# ---------- 平安人寿：一个 JSON 接口返回全部产品（含历史版本） ----------

PINGAN_PAGE = "https://life.pingan.com/gongkaixinxipilu/baoxianchanpinmulujitiaokuan.jsp"
PINGAN_LIST = "https://life.pingan.com/ilife-home/product/getProductList"
PINGAN_PDF = "https://life.pingan.com/ilife-home/product/getPlanClausePdf"


def pingan_life(f: Fetcher, progress: Callable[[str], None]) -> Iterator[Clause]:
    headers = {"Referer": PINGAN_PAGE}
    for flag, status in (("Y", ON_SALE), ("N", STOPPED)):
        body = {
            "isOrNotSale": flag,
            "planSalesStatus": flag,
            "sourceCode": "ilife-core",
            "planCode": "",
            "planDesc": "",
            # N：同一产品的历史版本也列出来
            "isOnlyNew": "N",
        }
        data = f.post(PINGAN_LIST, json=body, headers=headers).json()
        if data.get("CODE") != "00":
            raise RuntimeError(f"平安人寿接口返回 {data.get('CODE')}：{data.get('MSG')}")
        items = data.get("DATA") or []
        progress(f"{status} {len(items)} 条")
        for it in items:
            date = it.get("startDate" if flag == "Y" else "endDate") or ""
            # 页面脚本的规则：2017-04-01 之前的产品总有条款，之后的看 clauseContent
            if it.get("clauseContent") != "1" and not date < "2017-04-01":
                continue
            # 接口里的编号可能是数字，统一转成字符串，后面拼文件名、写 CSV 都按字符串处理
            plan, version = str(it["planCode"]), str(it["versionNo"])
            code = str(it.get("actualPlanCode") or plan)
            # 保留 /：组合产品的代码形如 866/866D，官网链接里就是原样的斜杠
            query = urlencode({"planCode": code, "versionNo": version, "attachmentType": 1}, safe="/")
            yield Clause(
                "pingan-life",
                str(it.get("clauseName") or it.get("planDesc") or code),
                status,
                version,
                f"{PINGAN_PDF}?{query}",
                product_id=plan,
                headers=headers,
            )


# ---------- 中国人寿：分页 JSON 接口，type=1 在售、2 停售 ----------

CHINALIFE_PAGE = "https://www.e-chinalife.com/xxpl/gsgk/cpjbxx/"
CHINALIFE_LIST = (
    "https://www.e-chinalife.com/jw/web/json/app/product_info_publish/plugin/"
    "com.chinalife.yunque.ProductInfoPublish/service"
)
CHINALIFE_PAGE_SIZE = 200


def chinalife_life(f: Fetcher, progress: Callable[[str], None]) -> Iterator[Clause]:
    headers = {"Referer": CHINALIFE_PAGE}
    for kind, status in ((1, ON_SALE), (2, STOPPED)):
        page, seen = 1, 0
        while True:
            params = {"pageSize": CHINALIFE_PAGE_SIZE, "page": page, "type": kind}
            data = f.get(CHINALIFE_LIST, params=params, headers=headers).json()
            # 没有 count 就没法判断翻到哪一页为止，按接口改版处理
            if not isinstance(data, dict) or "count" not in data:
                raise RuntimeError("中国人寿接口的返回里没有 count 字段，接口可能改版了")
            items = data.get("list") or []
            total = int(data.get("count") or 0)
            for it in items:
                clause = (it.get("clause") or "").strip()
                if clause and clause != "/":
                    yield Clause(
                        "chinalife-life",
                        it.get("productname") or "",
                        status,
                        "当前",
                        urljoin(CHINALIFE_PAGE, clause),
                        product_id=str(it.get("id") or it.get("serialno") or ""),
                    )
            seen += len(items)
            progress(f"{status} {seen}/{total}")
            if not items and seen < total:
                raise RuntimeError(f"中国人寿接口第 {page} 页是空的，但总数是 {total}，接口可能改版了")
            if not items or seen >= total:
                break
            page += 1


# ---------- 新华保险：服务端渲染的列表页，条款在每个产品的详情页里（含已停用的版本） ----------

NCL_BASE = "https://www.newchinalife.com"
NCL_LIST = NCL_BASE + "/node/372"
RE_NCL_PAGES = re.compile(r"共\s*(\d+)\s*页")
RE_NCL_CLAUSE = re.compile(r"<a[^>]+href=\"([^\"]+)\"[^>]*>\s*条款\s*</a>")


def _ncl_product_table(page: str) -> str:
    for table in re.findall(r"<table.*?</table>", page, re.S):
        if "产品名称" in table:
            return table
    return ""


RE_NCL_PDF = re.compile(r'href="([^"]+?\.pdf(?:\?[^"]*)?)"', re.I)


def _ncl_detail(f: Fetcher, url: str) -> list[tuple[str, str, str]]:
    page = f.get(url, headers={"Referer": NCL_LIST}).text
    start = page.find("tlInsurance")
    if start < 0:
        return []
    out = []
    for row in _rows(page[start : page.find("</table>", start)]):
        cells = _cells(row)
        link = RE_NCL_PDF.search(row)
        if link and len(cells) >= 2:
            stopped = _text(cells[2]) if len(cells) > 2 else ""
            href = urljoin(url, html.unescape(link.group(1)))
            out.append((_text(cells[1]), href, f"停用于 {stopped}" if stopped else "当前"))
    return out


def newchinalife(f: Fetcher, progress: Callable[[str], None]) -> Iterator[Clause | Skipped]:
    first = f.get(NCL_LIST).text
    pages = int(RE_NCL_PAGES.search(first).group(1)) if RE_NCL_PAGES.search(first) else 1
    for n in range(1, pages + 1):
        page = first if n == 1 else f.get(f"{NCL_LIST}_{n}", headers={"Referer": NCL_LIST}).text
        table = _ncl_product_table(page)
        if not table:
            raise RuntimeError(f"新华保险列表第 {n} 页里找不到产品表，页面结构可能变了")
        for row in _rows(table):
            cells = _cells(row)
            link = RE_NCL_CLAUSE.search(row)
            if len(cells) < 4 or not link:
                continue
            name, status = _text(cells[1]), _text(cells[3])
            href = html.unescape(link.group(1))
            risk = parse_qs(urlsplit(href).query).get("riskCode", [name])[0]
            if urlsplit(href).path.lower().endswith(".pdf"):
                found = [(name, urljoin(NCL_BASE, href), "当前")]
            else:
                # 八百多个详情页，个别取不到时只跳过这一个，不让整家的遍历作废
                detail = urljoin(NCL_BASE, href)
                try:
                    found = _ncl_detail(f, detail)
                except (httpx.HTTPError, Blocked) as exc:
                    yield Skipped(detail, f"{type(exc).__name__}: {exc}", product_id=risk)
                    continue
            for clause_name, url, version in found:
                yield Clause("newchinalife", clause_name or name, status, version, url, product_id=risk)
        progress(f"第 {n}/{pages} 页")


# ---------- 太保寿险：按在售 / 停售 × 险种分 8 个目录，服务端渲染分页 ----------

CPIC_BASE = "https://life.cpic.com.cn"
CPIC_ROOT = CPIC_BASE + "/xrsbx/gkxxpl/jbxx/gsgk/jydbxcpmljtk/"
CPIC_CATEGORIES = [f"{s}/{k}" for s in ("zbcp", "tbcp") for k in ("sx", "nj", "ywx", "jkx")]
RE_CPIC_PAGES = re.compile(r"ele\.value>(\d+)")
RE_CPIC_LINK = re.compile(r'href="([^"]+)"[^>]*title="([^"]*)"')


def _cpic_rows(page: str) -> list[tuple[str, str]]:
    start = page.find('id="example"')
    if start < 0:
        # 空目录也有这张表；连表都没有说明页面结构变了，不能当成 0 条
        raise RuntimeError('太保寿险目录页里找不到条款表（id="example"），页面结构可能变了')
    out = []
    for row in _rows(page[start : page.find("</table>", start)]):
        cells = _cells(row)
        link = RE_CPIC_LINK.search(cells[1]) if len(cells) >= 2 else None
        if not link:
            continue
        # 链接是 /mpdfjs/pdf.html?file=/upload/...pdf 的在线预览，取出 file 参数就是 PDF 本身
        href = html.unescape(link.group(1))
        path = href.split("file=", 1)[1].split("?", 1)[0] if "file=" in href else href
        out.append((html.unescape(link.group(2)), urljoin(CPIC_BASE, path)))
    return out


def cpic_life(f: Fetcher, progress: Callable[[str], None]) -> Iterator[Clause]:
    headers = {"Referer": CPIC_ROOT}
    for category in CPIC_CATEGORIES:
        status = ON_SALE if category.startswith("zbcp") else STOPPED
        first = f.get(CPIC_ROOT + category + "/", headers=headers).text
        found = RE_CPIC_PAGES.search(first)
        pages = int(found.group(1)) if found else 1
        for n in range(1, pages + 1):
            page = first if n == 1 else f.get(f"{CPIC_ROOT}{category}/index_{n}.shtml", headers=headers).text
            for name, url in _cpic_rows(page):
                yield Clause("cpic-life", name, status, "当前", url, product_id=name)
        progress(f"{category} {pages} 页")


@dataclass(frozen=True)
class Source:
    name: str
    label: str
    catalog: str
    walk: Callable[[Fetcher, Callable[[str], None]], Iterator[Clause | Skipped]]


SOURCES = {
    s.name: s
    for s in (
        Source("pingan-life", "平安人寿", PINGAN_PAGE, pingan_life),
        Source("chinalife-life", "中国人寿", CHINALIFE_PAGE, chinalife_life),
        Source("newchinalife", "新华保险", NCL_LIST, newchinalife),
        Source("cpic-life", "太保寿险", CPIC_ROOT, cpic_life),
    )
}
