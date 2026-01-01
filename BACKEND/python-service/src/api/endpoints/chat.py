import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ..core.models import ChatRequest
from ..services.query_service import QueryService
from .dependencies import get_query_service

router = APIRouter()

@router.post("/chat")
async def chat(
    request: ChatRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """Handle a chat request and stream the response."""
    if not request.history:
        raise HTTPException(status_code=400, detail="Chat history is required")

    async def response_stream():
        try:
            async for event in query_service.ask_stream(request.history, request.fileId):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")
