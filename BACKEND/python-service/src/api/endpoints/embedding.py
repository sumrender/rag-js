"""
API endpoints for generating embeddings.
"""
from fastapi import APIRouter, Depends, HTTPException
from ...core.models import EmbedTextRequest, EmbedTextResponse
from ...services.embedding_service import EmbeddingService
from ..dependencies import get_embedding_service

router = APIRouter()

@router.post("/embed-text", response_model=EmbedTextResponse)
async def embed_text(
    request: EmbedTextRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Generate text embedding for a single text string."""
    try:
        embedding_tuple = embedding_service.get_text_embedding(request.text)
        return {"embedding": list(embedding_tuple)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {e}")
