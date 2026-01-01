"""
API endpoints for file ingestion and processing.
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from ...core.models import (
    EmbedImagesBatchRequest, IngestFileRequest, IngestFileResponse
)
from ...services.file_processing_service import FileProcessingService
from ...services.ingestion_service import IngestionService
from ..dependencies import get_file_processing_service, get_ingestion_service

router = APIRouter()

@router.post("/extract-images")
async def extract_images(
    file: UploadFile = File(...),
    file_id: Optional[str] = Form(None),
    max_images: Optional[int] = Form(None),
    file_processing_service: FileProcessingService = Depends(get_file_processing_service),
):
    """Extract images from a PDF file."""
    return await file_processing_service.extract_images_from_pdf(file, file_id, max_images)

@router.post("/embed-images-batch")
async def embed_images_batch(
    request: EmbedImagesBatchRequest,
    file_processing_service: FileProcessingService = Depends(get_file_processing_service),
):
    """Generate CLIP embeddings for a batch of images."""
    return file_processing_service.embed_images_batch(request.image_urls, request.normalize)

@router.post("/ingest-file", response_model=IngestFileResponse)
async def ingest_file(
    request: IngestFileRequest,
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Trigger background ingestion job for a file."""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(
        ingestion_service.ingest_file,
        request.file_id,
        request.file_url,
        request.file_type,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "message": "Ingestion job started"},
    )

@router.get("/ingestion-status/{file_id}")
async def get_ingestion_status(
    file_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Get ingestion status for a file."""
    try:
        doc = ingestion_service.mongo_db.filemetadatas.find_one({"id": file_id})
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "readyForChatting": doc.get("readyForChatting", False),
            "stage": doc.get("ingestionStage", "unknown"),
            "error": doc.get("lastError"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
