from pathlib import Path as FsPath
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import FileResponse

from clausetrail.api.schemas import CHUNK_ID_PATTERN, DOC_ID_PATTERN

router = APIRouter(prefix="/documents", tags=["documents"])
DocId = Annotated[str, Path(pattern=DOC_ID_PATTERN)]
ChunkId = Annotated[str, Path(pattern=CHUNK_ID_PATTERN)]

PUBLIC_FIELDS = (
    "doc_id",
    "file_name",
    "pages",
    "chunks",
    "page_types",
    "parsers",
    "status",
    "kb_version",
    "ingested_at",
)
# 和入库支持的类型（ingest.router.IMAGE_SUFFIXES）对齐，浏览器才会预览而不是直接下载
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}


@router.get("")
def list_documents(request: Request) -> dict[str, Any]:
    db = request.app.state.services.db
    docs = [{k: d[k] for k in PUBLIC_FIELDS} for d in db.list_documents()]
    return {"kb_version": db.kb_version(), "documents": docs}


@router.get("/{doc_id}/chunks/{chunk_id}")
def get_chunk(doc_id: DocId, chunk_id: ChunkId, request: Request) -> dict[str, Any]:
    chunk = request.app.state.services.store.get_chunks([chunk_id]).get(chunk_id)
    if chunk is None or chunk.doc_id != doc_id:
        raise HTTPException(status_code=404, detail="片段不存在")
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "file": chunk.file_name,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
        "parser": chunk.parser,
        "kb_version": chunk.kb_version,
    }


@router.get("/{doc_id}/file")
def get_file(doc_id: DocId, request: Request) -> FileResponse:
    doc = request.app.state.services.db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    path = FsPath(doc["path"])
    if not path.is_file():
        raise HTTPException(status_code=410, detail="原文件已不在入库时的位置")
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=doc["file_name"],
        content_disposition_type="inline",
    )
