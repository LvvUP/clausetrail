"""公开条款语料的下载工具：robots.txt、按主机限速、重试、只保存 PDF。

对各家官网保持克制：
- 每个主机两次请求之间至少隔 delay 秒（默认 1 秒，不能低于 MIN_DELAY）；
- robots.txt 不允许的地址不请求。重定向自己逐跳跟随，每一跳都检查 robots.txt、按主机限速，
  不跟随从 https 降到 http 的跳转；
- robots.txt 返回 404 等 4xx 视为没有限制，401 / 403 按全部不允许；网络错误、5xx、429 重试后仍失败时，
  这个站点先按全部不允许处理，过 ROBOTS_RETRY_AFTER 秒再重新获取；
- 超时、5xx、408、429 退避后重试，429 带了 Retry-After 时按它等；
- 下载时先看文件头，不是 %PDF 就不保存；先写临时文件再改名，中断后不留半截文件，重跑时跳过已下载的。
"""

import logging
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "ClauseTrail-corpus/1.0 (research crawler for an open-source insurance clause QA demo)"
# README 承诺同一站点每秒最多一个请求，所以下限就是 1 秒
MIN_DELAY = 1.0
RETRIES = 3
MAX_REDIRECTS = 5
# robots.txt 取不到时先按全部不允许，过这么久再取一次：不让一次偶发超时封住整个站点，也不在站点故障时反复请求
ROBOTS_RETRY_AFTER = 300.0
MAX_RETRY_AFTER = 300.0
# 单个文件的上限，防止某个地址返回的不是条款而是超大文件
MAX_FILE_BYTES = 200 * 1024 * 1024


class Blocked(Exception):
    """robots.txt 不允许请求这个地址，或者重定向到了不跟随的地址。"""


class NotPdf(Exception):
    """返回的内容不是 PDF。"""


class _TryAgain(Exception):
    """5xx、408、429：退避后重试。"""

    def __init__(self, retry_after: float | None = None):
        super().__init__(retry_after)
        self.retry_after = retry_after


def _retryable(r: httpx.Response) -> bool:
    return r.status_code >= 500 or r.status_code in (408, 429)


def _retry_after(r: httpx.Response) -> float | None:
    # 只认秒数；HTTP 日期格式、负数、NaN 都忽略，按默认退避
    try:
        seconds = float(r.headers.get("retry-after", ""))
    except ValueError:
        return None
    return min(seconds, MAX_RETRY_AFTER) if seconds >= 0 else None


class Fetcher:
    def __init__(self, delay: float = 1.0, client: httpx.Client | None = None):
        self.delay = delay
        self._client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(30.0, connect=10.0)
        )
        self._last: dict[str, float] = {}
        # origin -> (规则, 过期时刻)；只有取不到 robots.txt 时才会过期
        self._robots: dict[str, tuple[RobotFileParser, float]] = {}

    def _rules(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        cached = self._robots.get(origin)
        if cached and time.monotonic() < cached[1]:
            return cached[0]

        def once() -> httpx.Response:
            # robots.txt 本身的跳转（如 http → https）也逐跳限速；跳到哪里，规则都用于原站点
            r = self._send("GET", origin + "/robots.txt", check_robots=False)
            if _retryable(r):
                raise _TryAgain(_retry_after(r))
            return r

        rp, expires = RobotFileParser(), float("inf")
        try:
            r = self._retry(origin + "/robots.txt", once)
        except (httpx.HTTPError, Blocked):
            log.warning("取不到 %s/robots.txt，这个站点先不请求，%.0f 秒后再试", origin, ROBOTS_RETRY_AFTER)
            rp.disallow_all = True
            expires = time.monotonic() + ROBOTS_RETRY_AFTER
        else:
            if r.status_code in (401, 403):
                rp.disallow_all = True
            elif r.status_code >= 400:
                rp.allow_all = True
            else:
                # 不看 content-type：有的站点把 robots.txt 标成 octet-stream。对不存在的地址返回 200 的 HTML 页面时，
                # 里面没有 User-agent / Disallow 指令，解析出来就是不限制
                rp.parse(r.text.splitlines())
        self._robots[origin] = (rp, expires)
        return rp

    def allowed(self, url: str) -> bool:
        return self._rules(url).can_fetch(USER_AGENT, url)

    def _throttle(self, host: str) -> None:
        wait = self._last.get(host, 0.0) + max(self.delay, 0.0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def _send(
        self, method: str, url: str, *, stream: bool = False, check_robots: bool = True, **kwargs
    ) -> httpx.Response:
        """发一次请求，自己逐跳跟随重定向：每一跳都按主机限速、检查 robots.txt（取 robots.txt 本身时不查）。"""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise httpx.UnsupportedProtocol(f"不是完整的 http(s) 地址：{url}")
        request = self._client.build_request(method, url, **kwargs)
        for _ in range(MAX_REDIRECTS + 1):
            here = str(request.url)
            if check_robots and not self.allowed(here):
                raise Blocked(here)
            self._throttle(urlsplit(here).netloc)
            r = self._client.send(request, stream=stream, follow_redirects=False)
            nxt = r.next_request
            if not r.is_redirect or nxt is None:
                return r
            r.close()
            if request.url.scheme == "https" and nxt.url.scheme != "https":
                raise Blocked(f"{here} 重定向到非 https 地址 {nxt.url}，不跟随")
            # 用 httpx 构造的下一跳：303 改成 GET、跨站时去掉认证头，这些规则和自动跟随时一致
            request = nxt
        raise httpx.TooManyRedirects(f"{url} 重定向超过 {MAX_REDIRECTS} 次", request=request)

    def _retry(self, url: str, once):
        for attempt in range(RETRIES):
            try:
                return once()
            except httpx.UnsupportedProtocol:
                raise
            except httpx.TransportError:
                if attempt == RETRIES - 1:
                    raise
                wait = None
            except _TryAgain as again:
                if attempt == RETRIES - 1:
                    break
                wait = again.retry_after
            # 退避 1、2 秒（delay 更大时按 delay 的倍数）；429 带了 Retry-After 时至少等这么久
            time.sleep(max(wait or 0.0, 2**attempt * max(self.delay, 1.0)))
        raise httpx.HTTPError(f"多次重试仍然失败：{url}")

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        def once():
            r = self._send(method, url, **kwargs)
            if _retryable(r):
                raise _TryAgain(_retry_after(r))
            r.raise_for_status()
            return r

        return self._retry(url, once)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self._request("POST", url, **kwargs)

    def download(self, url: str, dest: Path, headers: dict | None = None) -> int:
        """把 PDF 下载到 dest，返回字节数；dest 已经是 PDF 时不再下载，返回 0。"""
        if is_pdf_file(dest):
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        def once():
            r = self._send("GET", url, stream=True, headers=headers)
            try:
                if _retryable(r):
                    raise _TryAgain(_retry_after(r))
                r.raise_for_status()
                chunks = r.iter_bytes()
                first = next(chunks, b"")
                if not first.startswith(b"%PDF"):
                    raise NotPdf(f"{url} 返回的是 {r.headers.get('content-type', '未知类型')}，不是 PDF")
                size = len(first)
                with tmp.open("wb") as f:
                    f.write(first)
                    for chunk in chunks:
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise NotPdf(f"{url} 超过 {MAX_FILE_BYTES // 1024 // 1024} MB，不像条款文件")
                        f.write(chunk)
                return size
            finally:
                r.close()

        try:
            size = self._retry(url, once)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dest)
        return size

    def head(self, url: str, headers: dict | None = None) -> tuple[int, str, bytes]:
        """只读响应的第一块，不落盘，用来确认链接有效、内容是 PDF。"""

        def once():
            r = self._send("GET", url, stream=True, headers=headers)
            try:
                if _retryable(r):
                    raise _TryAgain(_retry_after(r))
                return r.status_code, r.headers.get("content-type", ""), next(r.iter_bytes(), b"")[:8]
            finally:
                r.close()

        return self._retry(url, once)

    def close(self) -> None:
        self._client.close()


def is_pdf_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        return f.read(4) == b"%PDF"


def free_gb(path: Path) -> float:
    path = path if path.exists() else next(p for p in path.parents if p.exists())
    return shutil.disk_usage(path).free / 1024**3
