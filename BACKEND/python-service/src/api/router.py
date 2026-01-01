"""
Central API router for the application.
"""
from fastapi import APIRouter
from .endpoints import ingestion, retrieval, embedding, utility, files, upload, chat

router = APIRouter()

router.include_router(ingestion.router, tags=["Ingestion"])
router.include_router(retrieval.router, tags=["Retrieval"])
router.include_router(embedding.router, tags=["Embedding"])
router.include_router(utility.router, tags=["Utility"])
router.include_router(files.router, tags=["Files"])
router.include_router(upload.router, tags=["Upload"])
router.include_router(chat.router, tags=["Chat"])
