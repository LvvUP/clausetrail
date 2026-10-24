import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
_USERINFO = re.compile(r"(?<=//)[^/@\s]+@")


def redact_url(url: str) -> str:
    """去掉连接串里的账号密码；写进日志和 /api/health 的地址只保留主机和端口。"""
    return _USERINFO.sub("***@", url)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    resources_dir: Path = BACKEND_ROOT / "resources"

    embedder_backend: Literal["aliyun_pai_eas", "local", "fake"] = "aliyun_pai_eas"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    aliyun_pai_eas_url: str = ""
    aliyun_pai_eas_token: str = ""

    reranker_backend: Literal["aliyun_opensearch", "local", "fake"] = "aliyun_opensearch"
    reranker_model: str = "bge-reranker-large"
    aliyun_opensearch_host: str = ""
    aliyun_opensearch_api_key: str = ""
    aliyun_opensearch_workspace: str = "default"
    aliyun_opensearch_ranker_id: str = "ops-bge-reranker-larger"
    reranker_max_tokens: int = 512

    llm_backend: Literal["aliyun_bailian", "fake"] = "aliyun_bailian"
    aliyun_bailian_api_key: str = ""
    aliyun_bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_rewrite_model: str = "qwen-flash"
    llm_answer_model: str = "qwen-plus"

    local_model_dir: Path = Path.home() / ".cache" / "clausetrail" / "models"

    # 不叫 MILVUS_URI：pymilvus 导入时会从环境变量和 .env 读取同名变量，且只接受 http(s) 地址
    vector_store_uri: str = str(PROJECT_ROOT / "data" / "milvus.db")
    vector_collection: str = "clause_chunks"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 128

    bm25_backend: Literal["es", "milvus"] = "es"
    es_url: str = "http://localhost:9200"
    es_index: str = "clause_chunks"
    es_username: str = ""
    es_password: str = ""

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    sqlite_path: Path = PROJECT_ROOT / "data" / "clausetrail.sqlite3"

    recall_top_k: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 20
    context_top_k: int = 5
    refusal_threshold: float = Field(0.35, description="在开发集上标定；低于此精排分直接拒答")

    chunk_max_tokens: int = 480
    chunk_overlap_sentences: int = 1
    merge_similarity_threshold: float = 0.82

    history_window_turns: int = 4
    summary_max_chars: int = 300
    continuity_similarity_threshold: float = 0.62

    layout_analysis: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # 只接受这些 Host：恶意网页把自己的域名解析到 127.0.0.1（DNS rebinding）时同源，CORS 拦不住，要靠 Host 校验
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "[::1]"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
