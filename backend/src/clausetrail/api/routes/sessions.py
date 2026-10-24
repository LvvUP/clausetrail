from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Request, Response

from clausetrail.api.schemas import SESSION_ID_PATTERN

router = APIRouter(prefix="/sessions", tags=["sessions"])
SessionId = Annotated[str, Path(pattern=SESSION_ID_PATTERN)]


@router.get("")
def list_sessions(request: Request) -> dict[str, Any]:
    return {"sessions": request.app.state.services.db.list_sessions()}


@router.get("/{session_id}")
def get_session(session_id: SessionId, request: Request) -> dict[str, Any]:
    db = request.app.state.services.db
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session": session, "turns": db.turns(session_id)}


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: SessionId, request: Request) -> Response:
    if not request.app.state.services.db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return Response(status_code=204)
