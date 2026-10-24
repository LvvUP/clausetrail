from typing import Any

from fastapi import APIRouter, Request

from clausetrail import __version__
from clausetrail.services import Services

router = APIRouter(tags=["health"])

CORE = ("embedder", "reranker", "llm")


def _model(services: Services, name: str) -> dict[str, Any]:
    component = getattr(services, name)
    if component is None:
        return {
            "status": "unavailable",
            "backend": getattr(services.settings, f"{name}_backend"),
            "detail": services.errors.get(name, ""),
        }
    return {"status": "ok", "backend": component.name}


def health_report(services: Services) -> dict[str, Any]:
    components = {name: _model(services, name) for name in CORE}
    try:
        vector_state = {"status": "ok", "index": services.store.index_type()}
    except Exception as exc:
        # 向量库连不上时只把这一项标成不可用，不让整个 health 接口返回 500
        vector_state = {"status": "unavailable", "detail": type(exc).__name__}
    components["vector_store"] = {
        "backend": "milvus_lite" if "://" not in services.store.uri else "milvus",
        "dim": services.store.dim,
        **vector_state,
    }
    components["bm25"] = {
        "status": "degraded" if "bm25" in services.degraded else "ok",
        "backend": services.bm25.name,
        "detail": services.degraded.get("bm25", ""),
    }
    components["cache"] = {
        "status": "degraded" if "cache" in services.degraded else "ok",
        "backend": services.cache.name,
        "detail": services.degraded.get("cache", ""),
    }
    ready = all(components[n]["status"] == "ok" for n in (*CORE, "vector_store"))
    return {
        "status": "ok" if ready else "unavailable",
        "version": __version__,
        "kb_version": services.db.kb_version(),
        "documents": services.db.count_documents(),
        "components": components,
    }


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    return health_report(request.app.state.services)
