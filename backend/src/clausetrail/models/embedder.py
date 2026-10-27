import asyncio
import hashlib
import logging
from functools import lru_cache
from typing import Protocol

import httpx
import numpy as np

from clausetrail.chunking.text_utils import tokenize_for_search
from clausetrail.config import Settings, get_settings
from clausetrail.models import want_fp16

log = logging.getLogger(__name__)

EMBED_BATCH = 16
# 几百个片段的文档不能一次把所有批次都发出去，容易被 EAS 限流；429 / 5xx 指数退避重试
EMBED_CONCURRENCY = 4
EMBED_RETRIES = 3
RETRY_BASE_SECONDS = 0.5
RETRY_STATUS = {429, 500, 502, 503, 504}


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


# 各后端都输出单位向量，下游（如 continuity.judge）直接用点积当余弦相似度
def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.clip(norms, 1e-12, None)).astype(np.float32)


class AliyunPaiEasEmbedder:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if not settings.aliyun_pai_eas_url or not settings.aliyun_pai_eas_token:
            raise ValueError("ALIYUN_PAI_EAS_URL / ALIYUN_PAI_EAS_TOKEN 未配置")
        self.name = f"aliyun_pai_eas:{settings.embedding_model}"
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model
        self._url = settings.aliyun_pai_eas_url.rstrip("/") + "/v1/embeddings"
        self._headers = {"Authorization": f"Bearer {settings.aliyun_pai_eas_token}"}
        self._client = client or httpx.AsyncClient(timeout=30)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(EMBED_RETRIES + 1):
            try:
                resp = await self._client.post(
                    self._url, headers=self._headers, json={"model": self._model, "input": texts}
                )
            except httpx.TransportError:
                if attempt == EMBED_RETRIES:
                    raise
            else:
                if resp.status_code not in RETRY_STATUS or attempt == EMBED_RETRIES:
                    resp.raise_for_status()
                    # 按返回项的 index 对回输入顺序，不依赖接口返回的先后
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    return [d["embedding"] for d in data]
            await asyncio.sleep(RETRY_BASE_SECONDS * 2**attempt)
        raise RuntimeError("unreachable")

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        limit = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def run(batch: list[str]) -> list[list[float]]:
            async with limit:
                return await self._batch(batch)

        results = await asyncio.gather(*(run(b) for b in batches))
        arr = np.asarray([v for r in results for v in r], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"向量维度是 {arr.shape[-1]}，与 EMBEDDING_DIM={self.dim} 不一致，请检查配置")
        return _normalize(arr)


class LocalBgeM3Embedder:
    def __init__(self, settings: Settings):
        from FlagEmbedding import BGEM3FlagModel

        path = settings.local_model_dir / "bge-m3"
        fp16 = want_fp16(settings.local_model_fp16)
        self.name = f"local:{path}" + (" (fp16)" if fp16 else "")
        self.dim = 1024
        self._model = BGEM3FlagModel(str(path), use_fp16=fp16)

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = await asyncio.to_thread(self._model.encode, texts, batch_size=EMBED_BATCH, max_length=1024)
        return _normalize(np.asarray(out["dense_vecs"], dtype=np.float32))


# 没有密钥和本地模型时用（测试即用它）：把词和相邻二字片段散列到固定维度，
# 用哈希最高位决定正负，抵消哈希冲突带来的偏差
class FakeEmbedder:
    def __init__(self, dim: int = 256):
        self.name = "fake:hashing"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        grams = tokenize_for_search(text) + [text[i : i + 2] for i in range(len(text) - 1)]
        for g in grams:
            h = int.from_bytes(hashlib.md5(g.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))


def build_embedder(settings: Settings) -> Embedder:
    match settings.embedder_backend:
        case "aliyun_pai_eas":
            return AliyunPaiEasEmbedder(settings)
        case "local":
            return LocalBgeM3Embedder(settings)
        case "fake":
            return FakeEmbedder()
    raise ValueError(f"未知的 embedder_backend: {settings.embedder_backend}")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return build_embedder(get_settings())


def embed_sync(embedder: Embedder, texts: list[str]) -> np.ndarray:
    return asyncio.run(embedder.embed(texts))
