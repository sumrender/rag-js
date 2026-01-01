"""
Main application entry point for the Python sidecar service.
"""
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.router import router as api_router
from .core.config import settings
from .api.dependencies import (
    get_embedding_service,
    get_text_store,
    get_image_store,
    get_mongo_client,
    get_cache_manager,
    get_semantic_cache,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal RAG Python Sidecar",
    version="1.0.0",
)

# Create images directory if it doesn't exist
images_dir = Path(settings.IMAGES_DIR)
images_dir.mkdir(exist_ok=True)

# Mount static files for image access
app.mount(f"/{settings.IMAGES_DIR}", StaticFiles(directory=images_dir), name="images")

# Include the main API router
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Preload models and connect to services on startup."""
    try:
        get_embedding_service().clip_model
        get_embedding_service().text_model
        get_text_store()
        get_image_store()
        get_mongo_client()
        get_cache_manager()
        get_semantic_cache()
        logger.info("Service startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize services on startup: {e}")
        logger.warning("Service will attempt to load on first request")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
