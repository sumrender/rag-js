from fastapi import APIRouter, Depends, HTTPException
from ..services.ingestion_service import IngestionService
from .dependencies import get_ingestion_service

router = APIRouter()

@router.get("/files")
async def get_files(
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Get all files from the database."""
    try:
        return ingestion_service.get_all_files()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
