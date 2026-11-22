import logging

from fastapi import APIRouter, HTTPException

from .schema import chat_payload
from . import service

logger = logging.getLogger(__name__)

ai_agent_router = APIRouter(prefix="/ai")


@ai_agent_router.post("/chat")
def initiate_chat(payload: chat_payload):
    if (not payload.query.strip() or not payload.session_id.strip()
            or any(char.isspace() for char in payload.session_id)):
        raise HTTPException(status_code=422, detail="Query and session ID must not be blank.")
    try:
        answer = service.chat(payload.query, payload.session_id)
    except Exception as error:
        # Provider failures are logged here and returned as a clean 502 instead
        # of an unhandled 500 that would confuse the frontend error boundary.
        logger.exception("Chat request failed for session %s", payload.session_id)
        raise HTTPException(
            status_code=502,
            detail="The assistant couldn't complete this request.",
        ) from error
    return {"session_id": payload.session_id, "answer": answer}


@ai_agent_router.get("/history")
def get_history(session_id: str):
    if not session_id.strip() or any(char.isspace() for char in session_id):
        raise HTTPException(status_code=422, detail="Session ID must not be blank.")
    return {"session_id": session_id, "messages": service.get_history(session_id)}
