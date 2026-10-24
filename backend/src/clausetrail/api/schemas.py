from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[0-9a-f]{16}$"
DOC_ID_PATTERN = r"^[0-9a-f]{16}$"
CHUNK_ID_PATTERN = r"^[0-9a-f]{16}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
