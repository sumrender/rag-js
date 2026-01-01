"""
API endpoints for utility functions.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from ...core.models import (
    SemanticCacheSearchRequest,
    SemanticCacheSearchResponse,
    SemanticCacheStoreRequest,
    SemanticCacheStoreResponse,
)
from ...core.config import settings
from ...services.embedding_service import EmbeddingService
from ...services.vector_store import FAISSStore
from ...services.cache_manager import RedisCacheManager, SemanticCache
from ..dependencies import (
    get_embedding_service,
    get_text_store,
    get_image_store,
    get_mongo_client,
    get_cache_manager,
    get_semantic_cache,
    get_retrieval_service,
    get_ingestion_service,
    get_file_processing_service,
)

router = APIRouter()

@router.get("/health")
async def health_check(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    text_store: FAISSStore = Depends(get_text_store),
    image_store: FAISSStore = Depends(get_image_store),
    mongo_client: MongoClient = Depends(get_mongo_client),
):
    """Health check endpoint."""
    try:
        # Check models
        clip_model_ready = embedding_service.clip_model is not None
        text_model_ready = embedding_service.text_model is not None

        # Check vector stores
        text_store_ready = text_store is not None and text_store.index is not None
        image_store_ready = image_store is not None and image_store.index is not None

        # Check MongoDB connection
        mongo_connected = False
        if mongo_client:
            try:
                mongo_client.admin.command('ping')
                mongo_connected = True
            except Exception:
                mongo_connected = False

        return {
            "status": "ok",
            "clip_model": settings.CLIP_MODEL_NAME if clip_model_ready else "not_loaded",
            "text_model": settings.TEXT_MODEL_NAME if text_model_ready else "not_loaded",
            "device": settings.DEVICE,
            "faiss_text_store_ready": text_store_ready,
            "faiss_images_store_ready": image_store_ready,
            "mongo_connected": mongo_connected,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@router.post("/reset-store")
async def reset_store(store_type: str = "all"):
    """Reset text, image, or all stores."""
    try:
        if store_type not in ["text", "image", "all"]:
            raise HTTPException(status_code=400, detail="store_type must be 'text', 'image', or 'all'")

        if store_type in ["text", "all"]:
            text_store = get_text_store()
            text_store.index = None
            if text_store.index_path.exists():
                text_store.index_path.unlink()
            if text_store.metadata_path.exists():
                text_store.metadata_path.unlink()
            if text_store.documents_path.exists():
                text_store.documents_path.unlink()
            if text_store.mapping_path.exists():
                text_store.mapping_path.unlink()

        if store_type in ["image", "all"]:
            image_store = get_image_store()
            image_store.index = None
            if image_store.index_path.exists():
                image_store.index_path.unlink()
            if image_store.metadata_path.exists():
                image_store.metadata_path.unlink()
            if image_store.documents_path.exists():
                image_store.documents_path.unlink()
            if image_store.mapping_path.exists():
                image_store.mapping_path.unlink()

        # Clear the caches of all services
        get_text_store.cache_clear()
        get_image_store.cache_clear()
        get_mongo_client.cache_clear()
        get_cache_manager.cache_clear()
        get_semantic_cache.cache_clear()
        get_file_processing_service.cache_clear()
        get_ingestion_service.cache_clear()
        get_retrieval_service.cache_clear()

        cache_manager = get_cache_manager()
        if cache_manager and store_type == "all":
            cache_manager.invalidate_all_cache()

        return {"message": f"Store(s) reset successfully: {store_type}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset store failed: {e}")


@router.post("/semantic-cache/search", response_model=SemanticCacheSearchResponse)
async def semantic_cache_search(
    request: SemanticCacheSearchRequest,
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
):
    """Search for similar cached query."""
    if not semantic_cache:
        return SemanticCacheSearchResponse(found=False)
    result = semantic_cache.search(request.query_embedding, request.file_id)
    if result:
        return SemanticCacheSearchResponse(found=True, **result)
    return SemanticCacheSearchResponse(found=False)


@router.post("/semantic-cache/store", response_model=SemanticCacheStoreResponse)
async def semantic_cache_store(
    request: SemanticCacheStoreRequest,
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
):
    """Store query-response pair in semantic cache."""
    if not semantic_cache:
        return SemanticCacheStoreResponse(success=False, message="Semantic cache is disabled")
    success = semantic_cache.store(
        request.query_embedding,
        request.response,
        request.query_text,
        request.file_id,
    )
    return SemanticCacheStoreResponse(success=success, message="Stored in semantic cache" if success else "Failed to store")


@router.post("/semantic-cache/clear")
async def semantic_cache_clear(
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
):
    """Clear all entries from semantic cache."""
    if not semantic_cache:
        return {"success": False, "message": "Semantic cache is disabled"}
    semantic_cache.clear()
    return {"success": True, "message": "Semantic cache cleared"}
