"""
API endpoints for retrieving text and images.
"""
from fastapi import APIRouter, Depends
from ...core.models import (
    RetrieveTextRequest, RetrieveTextResponse,
    RetrieveImagesByTextRequest, RetrieveImagesByTextResponse,
    RetrieveImagesByPagesRequest,
)
from ...services.retrieval_service import RetrievalService
from ..dependencies import get_retrieval_service

router = APIRouter()

@router.post("/retrieve-text", response_model=RetrieveTextResponse)
async def retrieve_text(
    request: RetrieveTextRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Retrieve text chunks by query."""
    return retrieval_service.retrieve_text(request.question, request.file_id, request.k)

@router.post("/retrieve-images-by-text", response_model=RetrieveImagesByTextResponse)
async def retrieve_images_by_text(
    request: RetrieveImagesByTextRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Retrieve images by text query."""
    return retrieval_service.retrieve_images_by_text(request.question, request.file_id, request.k)

@router.post("/retrieve-images-by-pages", response_model=RetrieveImagesByTextResponse)
async def retrieve_images_by_pages(
    request: RetrieveImagesByPagesRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Retrieve images by page numbers."""
    return retrieval_service.retrieve_images_by_pages(
        request.file_id, request.page_numbers, request.max_per_page
    )
