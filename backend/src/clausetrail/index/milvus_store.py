import json
import logging
import re
import threading
from functools import wraps
from pathlib import Path

import numpy as np
from pymilvus import DataType, Function, FunctionType, MilvusClient

from clausetrail.types import Chunk

log = logging.getLogger(__name__)

TEXT_MAX = 65535
INSERT_BATCH = 128
DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "file_name",
    "text",
    "section_path",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "parser",
    "kb_version",
    "bboxes",
]


# doc_id 会直接拼进 Milvus 的过滤表达式，先校验格式，防止表达式注入
def _check_doc_id(doc_id: str) -> str:
    if not DOC_ID_PATTERN.match(doc_id):
        raise ValueError(f"非法 doc_id: {doc_id!r}")
    return doc_id


# 检索和入库都经 asyncio.to_thread 调用，多个线程可能同时用同一个 client，这里统一串行；
# 用 RLock，加锁的方法之间互相调用时不会死锁
def _serialized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class MilvusStore:
    def __init__(self, uri: str, collection: str, dim: int, m: int = 16, ef_construction: int = 200, ef: int = 128):
        if "://" not in uri:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        self.uri = uri
        self.collection = collection
        self.dim = dim
        self._m, self._efc, self._ef = m, ef_construction, ef
        self._lock = threading.RLock()
        self._client = MilvusClient(uri)
        self._ensure_collection()

    def _vector_dim(self) -> int | None:
        for f in self._client.describe_collection(self.collection).get("fields", []):
            if f.get("name") == "vec":
                dim = f.get("params", {}).get("dim") or f.get("dim")
                return int(dim) if dim else None
        return None

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.collection):
            existing = self._vector_dim()
            if existing and existing != self.dim:
                raise RuntimeError(
                    f"集合 {self.collection} 的向量维度是 {existing}，当前向量模型是 {self.dim} 维；"
                    f"请删除 {self.uri} 后重新入库"
                )
            self._client.load_collection(self.collection)
            return
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX)
        schema.add_field("section_path", DataType.VARCHAR, max_length=4096)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("char_start", DataType.INT64)
        schema.add_field("char_end", DataType.INT64)
        schema.add_field("parser", DataType.VARCHAR, max_length=64)
        schema.add_field("kb_version", DataType.INT64)
        schema.add_field("bboxes", DataType.VARCHAR, max_length=TEXT_MAX)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(
            "tokens",
            DataType.VARCHAR,
            max_length=TEXT_MAX,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "standard"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        # tokens 存 jieba 预分词后的文本，sparse 由 BM25 Function 在写入时自动生成，插入时不用提供
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["tokens"],
                output_field_names=["sparse"],
            )
        )
        self._client.create_collection(self.collection, schema=schema)
        index = self._client.prepare_index_params()
        index.add_index(
            field_name="vec",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": self._m, "efConstruction": self._efc},
        )
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
        self._client.create_index(self.collection, index)
        self._client.load_collection(self.collection)
        log.info(
            "created milvus collection %s (dim=%d, HNSW M=%d efConstruction=%d)",
            self.collection,
            self.dim,
            self._m,
            self._efc,
        )

    @_serialized
    def index_type(self) -> str:
        return str(self._client.describe_index(self.collection, "vec").get("index_type", ""))

    @_serialized
    def delete_document(self, doc_id: str, keep_version: int | None = None, only_version: int | None = None) -> None:
        expr = f'doc_id == "{_check_doc_id(doc_id)}"'
        if keep_version is not None:
            expr += f" and kb_version != {int(keep_version)}"
        if only_version is not None:
            expr += f" and kb_version == {int(only_version)}"
        self._client.delete(self.collection, filter=expr)

    # 只插入不删除：重新入库时新版本先写进来，提交后再按版本删旧片段（见 ingest.pipeline），写到一半失败不丢旧数据
    @_serialized
    def insert_chunks(self, chunks: list[Chunk], vectors: np.ndarray, tokens: list[str]) -> int:
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "file_name": c.file_name,
                "text": c.text,
                "section_path": json.dumps(c.section_path, ensure_ascii=False),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "parser": c.parser,
                "kb_version": c.kb_version,
                "bboxes": json.dumps(c.bboxes),
                "vec": [float(x) for x in vec],
                "tokens": tok,
            }
            for c, vec, tok in zip(chunks, vectors, tokens, strict=True)
        ]
        for i in range(0, len(rows), INSERT_BATCH):
            self._client.insert(self.collection, rows[i : i + INSERT_BATCH])
        return len(rows)

    @_serialized
    def search_dense(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        res = self._client.search(
            self.collection,
            data=[[float(x) for x in vector]],
            anns_field="vec",
            limit=k,
            # HNSW 搜索要求 ef 不小于返回条数，k 较大时跟着调高
            search_params={"metric_type": "COSINE", "params": {"ef": max(self._ef, k)}},
            output_fields=["chunk_id"],
        )
        return [(h["chunk_id"], float(h["distance"])) for h in res[0]]

    @_serialized
    def search_bm25(self, query_tokens: str, k: int) -> list[tuple[str, float]]:
        if not query_tokens.strip():
            return []
        res = self._client.search(
            self.collection,
            data=[query_tokens],
            anns_field="sparse",
            limit=k,
            search_params={"metric_type": "BM25"},
            output_fields=["chunk_id"],
        )
        # milvus-lite 3.2.2 返回的 BM25 distance 是负的分数，取反后越大越相关，和 ES 的 _score 同向
        return [(h["chunk_id"], -float(h["distance"])) for h in res[0]]

    @_serialized
    def document_tokens(self, doc_id: str) -> list[tuple[str, str, str]]:
        """（chunk_id, doc_id, 预分词文本），用于重建 ES 索引。"""
        rows = self._client.query(
            self.collection,
            filter=f'doc_id == "{_check_doc_id(doc_id)}"',
            output_fields=["chunk_id", "doc_id", "tokens"],
            limit=16384,
        )
        return [(r["chunk_id"], r["doc_id"], r["tokens"]) for r in rows]

    @_serialized
    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        rows = self._client.get(self.collection, ids=list(chunk_ids), output_fields=OUTPUT_FIELDS)
        out = {}
        for r in rows:
            out[r["chunk_id"]] = Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                file_name=r["file_name"],
                text=r["text"],
                section_path=json.loads(r["section_path"]),
                page_start=int(r["page_start"]),
                page_end=int(r["page_end"]),
                char_start=int(r["char_start"]),
                char_end=int(r["char_end"]),
                parser=r["parser"],
                kb_version=int(r["kb_version"]),
                bboxes=[tuple(b) for b in json.loads(r["bboxes"])],
            )
        return out

    def close(self) -> None:
        self._client.close()
