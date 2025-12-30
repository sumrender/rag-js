"""
Python Sidecar Service for Multimodal RAG
Handles PDF image extraction, text/image embedding generation, FAISS vector storage, and MongoDB status updates
"""
import base64
import functools
import hashlib
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer
from faiss_store import FAISSStore
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from redis_cache import RedisCacheManager
from semantic_cache import SemanticCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG Python Sidecar", version="1.0.0")

# Images directory for storing extracted images
IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Global models (loaded once at startup)
clip_model: Optional[SentenceTransformer] = None
text_model: Optional[SentenceTransformer] = None
device: str = "cpu"
clip_model_name: str = "sentence-transformers/clip-ViT-B-32"
text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS stores
# Text embeddings: all-MiniLM-L6-v2 has 384 dimensions
# Image embeddings: clip-ViT-B-32 has 512 dimensions
text_store: Optional[FAISSStore] = None
images_store: Optional[FAISSStore] = None

# MongoDB client
mongo_client: Optional[MongoClient] = None
mongo_db = None

# Redis cache manager
cache_manager: Optional[RedisCacheManager] = None

# Semantic cache manager
semantic_cache: Optional[SemanticCache] = None


def get_clip_model() -> SentenceTransformer:
    """Load CLIP model (singleton pattern)"""
    global clip_model, device, clip_model_name
    
    if clip_model is None:
        clip_model_name = os.getenv("CLIP_MODEL_NAME", "sentence-transformers/clip-ViT-B-32")
        device = os.getenv("DEVICE", "cpu")
        logger.info(f"Loading CLIP model: {clip_model_name} on device: {device}")
        clip_model = SentenceTransformer(clip_model_name, device=device)
        logger.info("CLIP model loaded successfully")
    
    return clip_model


def get_text_model() -> SentenceTransformer:
    """Load text embedding model (singleton pattern)"""
    global text_model, device, text_model_name
    
    if text_model is None:
        text_model_name = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.getenv("DEVICE", "cpu")
        logger.info(f"Loading text model: {text_model_name} on device: {device}")
        text_model = SentenceTransformer(text_model_name, device=device)
        logger.info("Text model loaded successfully")
    
    return text_model


def get_text_store() -> FAISSStore:
    """Get or create text FAISS store"""
    global text_store
    
    if text_store is None:
        store_name = os.getenv("FAISS_TEXT_COLLECTION", "text_collection")
        data_dir = os.getenv("FAISS_DATA_DIR", "./faiss-data")
        # all-MiniLM-L6-v2 has 384 dimensions
        text_store = FAISSStore(store_name=store_name, dimension=384, data_dir=data_dir)
        logger.info(f"Text store '{store_name}' ready")
    
    return text_store


def get_images_store() -> FAISSStore:
    """Get or create images FAISS store"""
    global images_store
    
    if images_store is None:
        store_name = os.getenv("FAISS_IMAGE_COLLECTION", "images_collection")
        data_dir = os.getenv("FAISS_DATA_DIR", "./faiss-data")
        # clip-ViT-B-32 has 512 dimensions
        images_store = FAISSStore(store_name=store_name, dimension=512, data_dir=data_dir)
        logger.info(f"Images store '{store_name}' ready")
    
    return images_store


def get_mongo_db():
    """Get MongoDB database connection"""
    global mongo_client, mongo_db
    
    if mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "simple-rag")
        logger.info(f"Connecting to MongoDB at {mongo_uri}/{mongo_db_name}")
        mongo_client = MongoClient(mongo_uri)
        mongo_db = mongo_client[mongo_db_name]
        logger.info("MongoDB connected")
    
    return mongo_db


def get_cache_manager() -> Optional[RedisCacheManager]:
    """Get or create Redis cache manager"""
    global cache_manager
    
    if cache_manager is None:
        cache_manager = RedisCacheManager()
    
    return cache_manager


def get_semantic_cache() -> Optional[SemanticCache]:
    """Get or create semantic cache manager"""
    global semantic_cache
    
    if semantic_cache is None:
        enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
        if not enabled:
            return None
        
        dimension = 384  # all-MiniLM-L6-v2 dimension
        max_size = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "1000"))
        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
        semantic_cache = SemanticCache(dimension=dimension, max_size=max_size, threshold=threshold)
        logger.info("Semantic cache initialized")
    
    return semantic_cache


# Cached embedding functions using LRU cache
# Get cache size from environment variable, default 1000
_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _cached_text_embed(text: str) -> tuple:
    """
    Generate and cache text embedding (384D)
    
    Args:
        text: Input text string
    
    Returns:
        Tuple representation of embedding vector (for hashability)
    """
    model = get_text_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    # Convert numpy array to tuple for hashability (required by lru_cache)
    return tuple(embedding.tolist())


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _cached_clip_embed(text: str) -> tuple:
    """
    Generate and cache CLIP text embedding (512D)
    
    Args:
        text: Input text string
    
    Returns:
        Tuple representation of embedding vector (for hashability)
    """
    model = get_clip_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    # Convert numpy array to tuple for hashability (required by lru_cache)
    return tuple(embedding.tolist())


# Mount static files for image access
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

@app.on_event("startup")
async def startup_event():
    """Preload models and connect to services on startup"""
    try:
        get_clip_model()
        get_text_model()
        get_text_store()
        get_images_store()
        get_mongo_db()
        get_cache_manager()
        get_semantic_cache()
        logger.info("Service startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize services on startup: {e}")
        logger.warning("Service will attempt to load on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        clip_model = get_clip_model()
        text_model = get_text_model()
        text_store = get_text_store()
        images_store = get_images_store()
        mongo_db = get_mongo_db()
        
        return {
            "status": "ok",
            "clip_model": clip_model_name,
            "text_model": text_model_name,
            "device": device,
            "faiss_text_store_ready": text_store is not None,
            "faiss_images_store_ready": images_store is not None,
            "mongo_connected": mongo_db is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e)
            }
        )


def generate_image_id(file_id: str, page_num: int, xref: int, idx: int) -> str:
    """Generate stable unique image ID"""
    content = f"{file_id}|{page_num}|{xref}|{idx}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_image_id_from_url(image_url: str) -> str:
    """Extract image_id from image URL (e.g., '/images/abc123.png' -> 'abc123')"""
    # Remove /images/ prefix and .png suffix
    image_id = image_url.replace("/images/", "").replace(".png", "")
    return image_id


class EmbedImagesBatchRequest(BaseModel):
    """Request model for embed-images-batch endpoint"""
    file_id: str
    image_urls: List[str]
    normalize: bool = True


@app.post("/extract-images")
async def extract_images(
    file: UploadFile = File(...),
    file_id: Optional[str] = Form(None),
    max_images: Optional[int] = Form(None)
):
    """
    Extract images from PDF file
    
    Args:
        file: PDF file (multipart/form-data)
        file_id: Optional file identifier for correlation
        max_images: Optional maximum number of images to extract (safety guard)
    
    Returns:
        JSON with extracted images (base64 encoded) and metadata
    """
    warnings = []
    images_result = []
    
    # Use filename as file_id if not provided
    if file_id is None:
        file_id = file.filename or "unknown"
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        if not pdf_content:
            warnings.append("IMAGE_EXTRACTION_FAILED: Empty PDF file")
            return {
                "file_id": file_id,
                "images": [],
                "warnings": warnings
            }
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        total_images = 0
        for page_num in range(len(pdf_doc)):
            if max_images and total_images >= max_images:
                warnings.append(f"Reached max_images limit: {max_images}")
                break
            
            page = pdf_doc[page_num]
            
            # Extract images from page
            image_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(image_list):
                if max_images and total_images >= max_images:
                    break
                
                try:
                    xref = img_info[0]
                    
                    # Extract image data
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary (handles RGBA, L, etc.)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    
                    # Generate image ID
                    image_id = generate_image_id(file_id, page_num, xref, img_idx)
                    
                    # Get bounding box (best-effort)
                    # PyMuPDF doesn't always provide exact bbox for extracted images
                    # We'll try to get it from image placement on page
                    bbox = None
                    try:
                        # Try to find image placement on page
                        # This is approximate - actual bbox may not be available
                        image_rects = page.get_image_rects(xref)
                        if image_rects:
                            # Use first rect if available
                            rect = image_rects[0]
                            bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                    except Exception:
                        # Bbox not available - this is non-fatal
                        pass
                    
                    # Convert to base64
                    image_data = image_to_base64(pil_image, format="PNG")
                    
                    # Save image to disk
                    image_filename = f"{image_id}.png"
                    image_path = IMAGES_DIR / image_filename
                    pil_image.save(image_path, format="PNG")
                    
                    # Add image references
                    image_path_str = f"images/{image_filename}"
                    image_url = f"/images/{image_filename}"
                    
                    images_result.append({
                        "image_id": image_id,
                        "page_num": page_num,
                        "bbox": bbox,
                        # "image_data": image_data,
                        "image_path": image_path_str,
                        "image_url": image_url
                    })
                    
                    total_images += 1
                    
                except Exception as e:
                    warning_msg = f"Failed to extract image {img_idx} from page {page_num}: {str(e)}"
                    logger.warning(warning_msg)
                    warnings.append(f"PARTIAL_IMAGE_INGESTION: {warning_msg}")
                    continue
        
        pdf_doc.close()
        
        if not images_result and not warnings:
            warnings.append("IMAGE_EXTRACTION_FAILED: No images found in PDF")
        
        return {
            "file_id": file_id,
            "images": images_result,
            "warnings": warnings
        }
        
    except Exception as e:
        error_msg = f"IMAGE_EXTRACTION_FAILED: {str(e)}"
        logger.error(error_msg)
        warnings.append(error_msg)
        return {
            "file_id": file_id,
            "images": [],
            "warnings": warnings
        }


@app.post("/embed-images-batch")
async def embed_images_batch(request: EmbedImagesBatchRequest):
    """
    Generate CLIP embeddings for a batch of images (BATCH ONLY)
    
    Args:
        request: JSON body with file_id, image_urls array, and normalize flag
    
    Returns:
        JSON with embeddings array (image_id -> vector)
    """
    warnings = []
    embeddings_result = []
    
    # Handle empty input gracefully
    if not request.image_urls:
        return {
            "embeddings": [],
            "warnings": []
        }
    
    try:
        # Load CLIP model
        model = get_clip_model()
        
        # Process images and prepare for batch embedding
        pil_images = []
        valid_image_ids = []
        
        for image_url in request.image_urls:
            try:
                # Extract image_id from URL
                image_id = extract_image_id_from_url(image_url)
                
                # Construct file path
                image_path = IMAGES_DIR / f"{image_id}.png"
                
                # Validate file exists
                if not image_path.exists():
                    warnings.append(
                        f"IMAGE_EMBEDDING_FAILED: Image file not found for image_url: {image_url}"
                    )
                    continue
                
                # Load image from disk
                pil_image = Image.open(image_path)
                
                # Convert to RGB if necessary
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                
                pil_images.append(pil_image)
                valid_image_ids.append(image_id)
                
            except Exception as e:
                warning_msg = (
                    f"IMAGE_EMBEDDING_FAILED: Failed to process image {image_url}: {str(e)}"
                )
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                continue
        
        # Batch embed all valid images
        if pil_images:
            try:
                # Generate embeddings in single batch
                embeddings = model.encode(
                    pil_images,
                    convert_to_numpy=True,
                    normalize_embeddings=request.normalize,
                    batch_size=len(pil_images),  # Process all in one batch
                    show_progress_bar=False
                )
                
                # Convert to list format and pair with image_ids
                for image_id, embedding in zip(valid_image_ids, embeddings):
                    embeddings_result.append({
                        "image_id": image_id,
                        "embedding": embedding.tolist()
                    })
                
            except Exception as e:
                error_msg = f"PARTIAL_IMAGE_INGESTION: Batch embedding failed: {str(e)}"
                logger.error(error_msg)
                warnings.append(error_msg)
        else:
            warnings.append("PARTIAL_IMAGE_INGESTION: No valid images to embed")
        
        return {
            "embeddings": embeddings_result,
            "warnings": warnings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"IMAGE_EMBEDDING_FAILED: Unexpected error: {str(e)}"
        logger.error(error_msg)
        return {
            "embeddings": [],
            "warnings": [error_msg]
        }


class EmbedTextRequest(BaseModel):
    """Request model for embed-text endpoint"""
    text: str


class EmbedTextResponse(BaseModel):
    """Response model for embed-text endpoint"""
    embedding: List[float]


@app.post("/embed-text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest):
    """
    Generate text embedding for a single text string (used for query-time embeddings)
    
    Args:
        request: JSON body with text string
    
    Returns:
        JSON with embedding vector
    """
    try:
        # Use cached embedding function
        embedding_tuple = _cached_text_embed(request.text)
        # Convert tuple back to list for JSON response
        embedding = list(embedding_tuple)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {str(e)}")


class IngestFileRequest(BaseModel):
    """Request model for ingest-file endpoint"""
    file_id: str
    file_url: str
    file_type: Optional[str] = None  # 'pdf' or 'txt', auto-detected if not provided


class IngestFileResponse(BaseModel):
    """Response model for ingest-file endpoint"""
    job_id: str
    message: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk text with overlap (mirrors JS ChunkingService logic)"""
    chunks = []
    start = 0
    safe_overlap = min(chunk_overlap, chunk_size - 1)
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "startChar": start,
                "endChar": end
            })
        
        next_start = end - safe_overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
        
        if len(chunks) > 1000000:
            raise ValueError("Too many chunks generated")
    
    return chunks


def chunk_with_pages(pages: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk text with page tracking for PDFs (mirrors JS ChunkingService logic)"""
    chunks = []
    
    # Build character-to-page mapping
    current_char = 0
    char_to_page = []
    
    for page in pages:
        page_length = len(page["text"])
        for i in range(page_length):
            char_to_page.append({"char": current_char + i, "pageNum": page["pageNum"]})
        current_char += page_length
    
    # Combine all text
    full_text = "".join(p["text"] for p in pages)
    
    # Chunk with page tracking
    start = 0
    safe_overlap = min(chunk_overlap, chunk_size - 1)
    
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end].strip()
        
        if chunk_text:
            start_page = char_to_page[min(start, len(char_to_page) - 1)]["pageNum"] if char_to_page else 1
            end_page = char_to_page[min(end - 1, len(char_to_page) - 1)]["pageNum"] if char_to_page else 1
            
            chunk_data = {
                "text": chunk_text,
                "startChar": start,
                "endChar": end
            }
            
            if start_page == end_page:
                chunk_data["pageNumber"] = str(start_page)
            else:
                chunk_data["pageRange"] = f"{start_page}-{end_page}"
            
            chunks.append(chunk_data)
        
        next_start = end - safe_overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
        
        if len(chunks) > 1000000:
            raise ValueError("Too many chunks generated")
    
    return chunks


async def ingest_file_background(file_id: str, file_url: str, file_type: Optional[str] = None):
    """Background job for file ingestion"""
    warnings = []
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    try:
        logger.info(f"Starting ingestion for file_id={file_id}, file_url={file_url}")
        
        # Update MongoDB status
        db = get_mongo_db()
        file_metadata_collection = db["filemetadatas"]
        file_metadata_collection.update_one(
            {"id": file_id},
            {"$set": {"readyForChatting": False, "ingestionStage": "downloading"}}
        )
        
        # Download file
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content
        
        # Determine file type
        if file_type is None:
            if file_url.lower().endswith('.pdf'):
                file_type = 'pdf'
            elif file_url.lower().endswith('.txt'):
                file_type = 'txt'
            else:
                file_type = 'txt'  # default
        
        logger.info(f"File type detected: {file_type}")
        
        # Extract text
        text_chunks = []
        pages = None
        
        if file_type == 'pdf':
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            pages = []
            full_text = ""
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_text = page.get_text()
                pages.append({"text": page_text, "pageNum": page_num + 1})
                full_text += page_text
            
            pdf_doc.close()
            text_chunks = chunk_with_pages(pages, chunk_size, chunk_overlap)
        else:
            text = file_content.decode('utf-8', errors='ignore')
            text_chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        
        # Update status
        file_metadata_collection.update_one(
            {"id": file_id},
            {"$set": {"ingestionStage": "embedding_text"}}
        )
        
        # Batch embed text chunks
        text_model = get_text_model()
        chunk_texts = [chunk["text"] for chunk in text_chunks]
        
        if chunk_texts:
            text_embeddings = text_model.encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            
            # Prepare FAISS data
            text_store = get_text_store()
            chunk_ids = [f"{file_id}-chunk-{i}" for i in range(len(text_chunks))]
            chunk_metadatas = []
            
            for i, chunk in enumerate(text_chunks):
                metadata = {
                    "fileId": file_id,
                    "contentType": "chunk",
                    "chunkIndex": i,
                    "totalChunks": len(text_chunks),
                    "characterRange": f"{chunk['startChar']}-{chunk['endChar']}"
                }
                if "pageNumber" in chunk:
                    metadata["pageNumber"] = chunk["pageNumber"]
                if "pageRange" in chunk:
                    metadata["pageRange"] = chunk["pageRange"]
                chunk_metadatas.append(metadata)
            
            # Upsert to FAISS
            text_store.upsert(
                ids=chunk_ids,
                embeddings=text_embeddings.tolist(),
                documents=[chunk["text"] for chunk in text_chunks],
                metadatas=chunk_metadatas
            )
            logger.info(f"Upserted {len(text_chunks)} text chunks to FAISS")
        
        # Extract and embed images (PDF only)
        image_count = 0
        if file_type == 'pdf':
            try:
                file_metadata_collection.update_one(
                    {"id": file_id},
                    {"$set": {"ingestionStage": "extracting_images"}}
                )
                
                # Extract images
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                extracted_images = []
                
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    image_list = page.get_images(full=True)
                    
                    for img_idx, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = pdf_doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")
                            
                            image_id = generate_image_id(file_id, page_num, xref, img_idx)
                            image_filename = f"{image_id}.png"
                            image_path = IMAGES_DIR / image_filename
                            pil_image.save(image_path, format="PNG")
                            
                            # Get bbox if available
                            bbox = None
                            try:
                                image_rects = page.get_image_rects(xref)
                                if image_rects:
                                    rect = image_rects[0]
                                    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                            except Exception:
                                pass
                            
                            extracted_images.append({
                                "image_id": image_id,
                                "page_num": page_num,
                                "bbox": bbox,
                                "image_path": f"images/{image_filename}",
                                "image_url": f"/images/{image_filename}"
                            })
                        except Exception as e:
                            warning_msg = f"Failed to extract image {img_idx} from page {page_num}: {str(e)}"
                            logger.warning(warning_msg)
                            warnings.append(f"PARTIAL_IMAGE_INGESTION: {warning_msg}")
                
                pdf_doc.close()
                
                if extracted_images:
                    file_metadata_collection.update_one(
                        {"id": file_id},
                        {"$set": {"ingestionStage": "embedding_images"}}
                    )
                    
                    # Batch embed images
                    clip_model = get_clip_model()
                    image_paths = [IMAGES_DIR / f"{img['image_id']}.png" for img in extracted_images]
                    pil_images = []
                    valid_indices = []
                    
                    for i, img_path in enumerate(image_paths):
                        if img_path.exists():
                            pil_image = Image.open(img_path)
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")
                            pil_images.append(pil_image)
                            valid_indices.append(i)
                    
                    if pil_images:
                        image_embeddings = clip_model.encode(
                            pil_images,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=len(pil_images),
                            show_progress_bar=False
                        )
                        
                        # Prepare FAISS data for images
                        images_store = get_images_store()
                        image_ids = [extracted_images[i]["image_id"] for i in valid_indices]
                        image_metadatas = []
                        
                        for i in valid_indices:
                            img = extracted_images[i]
                            metadata = {
                                "fileId": file_id,
                                "pageNumber": img["page_num"] + 1,  # Convert to 1-indexed to match text chunks
                                "imageId": img["image_id"],
                                "imageUrl": img["image_url"]
                            }
                            if img["bbox"]:
                                metadata["bbox"] = json.dumps(img["bbox"])
                            
                            # Add nearbyText from page text (first 500 chars)
                            if pages is not None:
                                page_num_1_indexed = img["page_num"] + 1
                                # Find matching page in pages array
                                for page in pages:
                                    if page.get("pageNum") == page_num_1_indexed:
                                        page_text = page.get("text", "")
                                        # Cap at 500 characters
                                        metadata["nearbyText"] = page_text[:500]
                                        break
                            
                            image_metadatas.append(metadata)
                        
                        # Upsert to FAISS
                        images_store.upsert(
                            ids=image_ids,
                            embeddings=image_embeddings.tolist(),
                            documents=[img["image_id"] for img in [extracted_images[i] for i in valid_indices]],  # Use image_id as document
                            metadatas=image_metadatas
                        )
                        image_count = len(image_ids)
                        logger.info(f"Upserted {image_count} image embeddings to FAISS")
                
            except Exception as e:
                error_msg = f"IMAGE_EXTRACTION_FAILED: {str(e)}"
                logger.warning(error_msg)
                warnings.append(error_msg)
        
        # Update MongoDB status to ready
        file_metadata_collection.update_one(
            {"id": file_id},
            {
                "$set": {
                    "readyForChatting": True,
                    "ingestionStage": "complete",
                    "imageCount": image_count
                }
            }
        )
        
        # Invalidate cache for this file after successful ingestion
        cache_mgr = get_cache_manager()
        if cache_mgr:
            deleted_count = cache_mgr.invalidate_file_cache(file_id)
            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} Redis cache entries for file_id={file_id}")
        
        # Invalidate semantic cache for this file
        semantic_cache = get_semantic_cache()
        if semantic_cache:
            semantic_removed = semantic_cache.invalidate_file(file_id)
            if semantic_removed > 0:
                logger.info(f"Invalidated {semantic_removed} semantic cache entries for file_id={file_id}")
        
        logger.info(f"Ingestion complete for file_id={file_id}, chunks={len(text_chunks)}, images={image_count}")
        if warnings:
            logger.warning(f"Warnings during ingestion: {warnings}")
        
    except Exception as e:
        error_msg = f"Ingestion failed for file_id={file_id}: {str(e)}"
        logger.error(error_msg)
        
        # Update MongoDB status with error
        try:
            db = get_mongo_db()
            file_metadata_collection = db["filemetadatas"]
            file_metadata_collection.update_one(
                {"id": file_id},
                {
                    "$set": {
                        "readyForChatting": False,
                        "ingestionStage": "error",
                        "lastError": str(e)
                    }
                }
            )
        except Exception as mongo_error:
            logger.error(f"Failed to update MongoDB error status: {mongo_error}")


@app.post("/ingest-file", response_model=IngestFileResponse)
async def ingest_file(request: IngestFileRequest, background_tasks: BackgroundTasks):
    """
    Trigger background ingestion job for a file
    
    Args:
        request: JSON body with file_id, file_url, and optional file_type
        background_tasks: FastAPI background tasks
    
    Returns:
        202 Accepted with job_id
    """
    job_id = str(uuid.uuid4())
    
    # Start background ingestion
    background_tasks.add_task(
        ingest_file_background,
        request.file_id,
        request.file_url,
        request.file_type
    )
    
    logger.info(f"Started ingestion job {job_id} for file_id={request.file_id}")
    
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "message": "Ingestion job started"
        }
    )


@app.get("/ingestion-status/{file_id}")
async def get_ingestion_status(file_id: str):
    """
    Get ingestion status for a file
    
    Args:
        file_id: File identifier
    
    Returns:
        JSON with readyForChatting, stage, and optional error
    """
    try:
        db = get_mongo_db()
        file_metadata_collection = db["filemetadatas"]
        doc = file_metadata_collection.find_one({"id": file_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "readyForChatting": doc.get("readyForChatting", False),
            "stage": doc.get("ingestionStage", "unknown"),
            "error": doc.get("lastError")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RetrieveTextRequest(BaseModel):
    """Request model for retrieve-text endpoint"""
    question: str
    file_id: Optional[str] = None
    k: int = 3


class RetrieveTextResponse(BaseModel):
    """Response model for retrieve-text endpoint"""
    chunks: List[Dict[str, Any]]
    queryModel: str


@app.post("/retrieve-text", response_model=RetrieveTextResponse)
async def retrieve_text(request: RetrieveTextRequest):
    """
    Retrieve text chunks by query (Python queries FAISS)
    
    Args:
        request: JSON body with question, optional file_id, and k (number of results)
    
    Returns:
        JSON with chunks array and query model name
    """
    try:
        cache_mgr = get_cache_manager()
        
        # Phase 2.1: Check query results cache first
        if cache_mgr and os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true":
            cached_query_result = cache_mgr.get_query_cache(
                request.question, 
                request.file_id, 
                "text"
            )
            if cached_query_result:
                logger.info(f"Cache HIT for query results: '{request.question[:50]}...'")
                return {
                    "chunks": cached_query_result.get("chunks", []),
                    "queryModel": cached_query_result.get("queryModel", text_model_name)
                }
        
        # Generate query embedding using cache
        embedding_tuple = _cached_text_embed(request.question)
        # Convert tuple back to numpy array for FAISS query
        query_embedding = np.array(embedding_tuple, dtype=np.float32)
        
        # Phase 2.2: Check FAISS results cache
        faiss_results = None
        if cache_mgr and os.getenv("ENABLE_FAISS_CACHE", "true").lower() == "true":
            faiss_results = cache_mgr.get_faiss_cache(
                query_embedding,
                request.file_id,
                request.k,
                "text"
            )
            if faiss_results:
                logger.info(f"Cache HIT for FAISS results: '{request.question[:50]}...'")
        
        # If no FAISS cache hit, perform FAISS query
        if faiss_results is None:
            # Query FAISS text store
            text_store = get_text_store()
            where_filter = None
            if request.file_id:
                where_filter = {"fileId": {"$eq": request.file_id}}
            
            logger.info(f"Querying text store with question: '{request.question[:50]}...', k={request.k}, file_id={request.file_id}")
            faiss_results = text_store.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=request.k,
                where=where_filter
            )
            
            # Cache FAISS results
            if cache_mgr and os.getenv("ENABLE_FAISS_CACHE", "true").lower() == "true":
                cache_mgr.set_faiss_cache(
                    query_embedding,
                    request.file_id,
                    request.k,
                    "text",
                    faiss_results
                )
        
        logger.info(f"Query returned {len(faiss_results.get('documents', [[]])[0]) if faiss_results.get('documents') else 0} chunks")
        
        # Format results
        chunks = []
        if faiss_results.get("documents") and faiss_results["documents"][0]:
            documents = faiss_results["documents"][0]
            metadatas = faiss_results.get("metadatas", [[]])[0] or []
            distances = faiss_results.get("distances", [[]])[0] or []
            ids = faiss_results.get("ids", [[]])[0] or []
            
            for i, doc in enumerate(documents):
                if doc:
                    # Convert L2 distance to cosine similarity for normalized embeddings
                    # For normalized vectors: cosine_similarity = 1 - (squared_L2_distance / 2)
                    # FAISS IndexFlatL2 returns squared L2 distances
                    distance = distances[i] if i < len(distances) else 2.0
                    cosine_similarity = 1.0 - (distance / 2.0)
                    chunk_data = {
                        "id": ids[i] if i < len(ids) else None,
                        "text": doc,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "score": cosine_similarity
                    }
                    chunks.append(chunk_data)
        
        result = {
            "chunks": chunks,
            "queryModel": text_model_name
        }
        
        # Phase 2.1: Cache query results
        if cache_mgr and os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true":
            cache_mgr.set_query_cache(
                request.question,
                request.file_id,
                "text",
                result
            )
        
        return result
    except Exception as e:
        logger.error(f"Text retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text retrieval failed: {str(e)}")


class RetrieveImagesByTextRequest(BaseModel):
    """Request model for retrieve-images-by-text endpoint"""
    question: str
    file_id: Optional[str] = None
    k: int = 3


class RetrieveImagesByTextResponse(BaseModel):
    """Response model for retrieve-images-by-text endpoint"""
    images: List[Dict[str, Any]]


@app.post("/retrieve-images-by-text", response_model=RetrieveImagesByTextResponse)
async def retrieve_images_by_text(request: RetrieveImagesByTextRequest):
    """
    Retrieve images by text query (CLIP text-to-image search)
    
    Args:
        request: JSON body with question, optional file_id, and k (number of results)
    
    Returns:
        JSON with images array
    """
    try:
        cache_mgr = get_cache_manager()
        
        # Phase 2.1: Check query results cache first
        if cache_mgr and os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true":
            cached_query_result = cache_mgr.get_query_cache(
                request.question,
                request.file_id,
                "images"
            )
            if cached_query_result:
                logger.info(f"Cache HIT for image query results: '{request.question[:50]}...'")
                return {"images": cached_query_result.get("images", [])}
        
        # Generate CLIP text embedding using cache
        embedding_tuple = _cached_clip_embed(request.question)
        # Convert tuple back to numpy array for FAISS query
        query_embedding = np.array(embedding_tuple, dtype=np.float32)
        
        # Phase 2.2: Check FAISS results cache
        faiss_results = None
        if cache_mgr and os.getenv("ENABLE_FAISS_CACHE", "true").lower() == "true":
            faiss_results = cache_mgr.get_faiss_cache(
                query_embedding,
                request.file_id,
                request.k,
                "img"
            )
            if faiss_results:
                logger.info(f"Cache HIT for image FAISS results: '{request.question[:50]}...'")
        
        # If no FAISS cache hit, perform FAISS query
        if faiss_results is None:
            # Query FAISS images store
            images_store = get_images_store()
            where_filter = None
            if request.file_id:
                where_filter = {"fileId": {"$eq": request.file_id}}
            
            faiss_results = images_store.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=request.k,
                where=where_filter
            )
            
            # Cache FAISS results
            if cache_mgr and os.getenv("ENABLE_FAISS_CACHE", "true").lower() == "true":
                cache_mgr.set_faiss_cache(
                    query_embedding,
                    request.file_id,
                    request.k,
                    "img",
                    faiss_results
                )
        
        # Format results
        images = []
        if faiss_results.get("documents") and faiss_results["documents"][0]:
            documents = faiss_results["documents"][0]
            metadatas = faiss_results.get("metadatas", [[]])[0] or []
            distances = faiss_results.get("distances", [[]])[0] or []
            ids = faiss_results.get("ids", [[]])[0] or []
            
            for i, doc in enumerate(documents):
                if doc:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    # Convert L2 distance to cosine similarity for normalized embeddings
                    # For normalized vectors: cosine_similarity = 1 - (squared_L2_distance / 2)
                    # FAISS IndexFlatL2 returns squared L2 distances
                    distance = distances[i] if i < len(distances) else 2.0
                    cosine_similarity = 1.0 - (distance / 2.0)
                    image_data = {
                        "imageId": metadata.get("imageId", ids[i] if i < len(ids) else None),
                        "paintingId": metadata.get("paintingId", metadata.get("imageId")),  # Fallback to imageId if no paintingId
                        "fileId": metadata.get("fileId"),
                        "pageNumber": metadata.get("pageNumber"),
                        "imageUrl": metadata.get("imageUrl"),
                        "score": cosine_similarity,
                        "nearbyText": metadata.get("nearbyText")
                    }
                    images.append(image_data)
        
        result = {"images": images}
        
        # Phase 2.1: Cache query results
        if cache_mgr and os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true":
            cache_mgr.set_query_cache(
                request.question,
                request.file_id,
                "images",
                result
            )
        
        return result
    except Exception as e:
        logger.error(f"Image retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image retrieval failed: {str(e)}")


class RetrieveImagesByPagesRequest(BaseModel):
    """Request model for retrieve-images-by-pages endpoint"""
    file_id: Optional[str] = None
    page_numbers: List[int]  # 1-indexed
    max_per_page: int = 3


@app.post("/retrieve-images-by-pages", response_model=RetrieveImagesByTextResponse)
async def retrieve_images_by_pages(request: RetrieveImagesByPagesRequest):
    """
    Retrieve images by page numbers (deterministic, no similarity search)
    
    Args:
        request: JSON body with page_numbers list, optional file_id, and max_per_page
    
    Returns:
        JSON with images array
    """
    try:
        if not request.page_numbers:
            return {"images": []}
        
        # Query images store - get all images and filter in Python
        images_store = get_images_store()
        
        if images_store.index is None or images_store.index.ntotal == 0:
            return {"images": []}
        
        # Query all images with a dummy embedding (we'll filter by metadata)
        # Use a large n_results to get all images
        dummy_embedding = [[0.0] * 512]  # CLIP dimension is 512
        results = images_store.query(
            query_embeddings=dummy_embedding,
            n_results=min(images_store.index.ntotal, 10000),  # Cap at 10k for safety
            where={"fileId": {"$eq": request.file_id}} if request.file_id else None
        )
        
        # Format results and filter by page numbers
        images = []
        page_counts = {page_num: 0 for page_num in request.page_numbers}
        
        if results.get("documents") and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0] or []
            ids = results.get("ids", [[]])[0] or []
            
            for i, doc in enumerate(documents):
                if doc:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    page_num = metadata.get("pageNumber")
                    
                    # Filter by page numbers
                    if page_num in request.page_numbers:
                        # Apply max_per_page limit
                        if page_counts[page_num] >= request.max_per_page:
                            continue
                        
                        page_counts[page_num] += 1
                        
                        image_data = {
                            "imageId": metadata.get("imageId", ids[i] if i < len(ids) else None),
                            "paintingId": metadata.get("paintingId", metadata.get("imageId")),
                            "fileId": metadata.get("fileId"),
                            "pageNumber": page_num,
                            "imageUrl": metadata.get("imageUrl"),
                            "score": 1.0,  # No similarity score for page-linked retrieval
                            "nearbyText": metadata.get("nearbyText")
                        }
                        images.append(image_data)
        
        return {"images": images}
    except Exception as e:
        logger.error(f"Page-linked image retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Page-linked image retrieval failed: {str(e)}")


@app.post("/reset-store")
async def reset_store(store_type: str = "all"):
    """
    Reset text, image, or all stores (delete FAISS indices and metadata)
    
    Args:
        store_type: "text", "image", or "all" (default: "all")
    
    Returns:
        JSON with success message
    """
    global text_store, images_store
    
    try:
        if store_type not in ["text", "image", "all"]:
            raise HTTPException(status_code=400, detail="store_type must be 'text', 'image', or 'all'")
        
        if store_type in ["text", "all"]:
            store = get_text_store()
            # Delete files
            if store.index_path.exists():
                store.index_path.unlink()
            if store.metadata_path.exists():
                store.metadata_path.unlink()
            if store.documents_path.exists():
                store.documents_path.unlink()
            if store.mapping_path.exists():
                store.mapping_path.unlink()
            # Reset global variable to force recreation
            text_store = None
            logger.info("Text store reset")
        
        if store_type in ["image", "all"]:
            store = get_images_store()
            # Delete files
            if store.index_path.exists():
                store.index_path.unlink()
            if store.metadata_path.exists():
                store.metadata_path.unlink()
            if store.documents_path.exists():
                store.documents_path.unlink()
            if store.mapping_path.exists():
                store.mapping_path.unlink()
            # Reset global variable to force recreation
            images_store = None
            logger.info("Images store reset")
        
        # Invalidate entire cache when stores are reset
        cache_mgr = get_cache_manager()
        if cache_mgr and store_type == "all":
            cache_mgr.invalidate_all_cache()
            logger.info("Cache invalidated after store reset")
        
        return {
            "message": f"Store(s) reset successfully: {store_type}",
            "store_type": store_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset store failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset store failed: {str(e)}")



# Semantic Cache Endpoints

class SemanticCacheSearchRequest(BaseModel):
    """Request model for semantic cache search"""
    query_embedding: List[float]
    file_id: Optional[str] = None


class SemanticCacheSearchResponse(BaseModel):
    """Response model for semantic cache search"""
    found: bool
    response: Optional[str] = None
    similarity: Optional[float] = None
    query: Optional[str] = None
    file_id: Optional[str] = None


class SemanticCacheStoreRequest(BaseModel):
    """Request model for semantic cache store"""
    query_embedding: List[float]
    response: str
    query_text: str = ""
    file_id: Optional[str] = None


class SemanticCacheStoreResponse(BaseModel):
    """Response model for semantic cache store"""
    success: bool
    message: str


@app.post("/semantic-cache/search", response_model=SemanticCacheSearchResponse)
async def semantic_cache_search(request: SemanticCacheSearchRequest):
    """
    Search for similar cached query
    
    Args:
        request: JSON body with query_embedding and optional file_id
    
    Returns:
        Cached response if similarity > threshold, otherwise null
    """
    try:
        cache = get_semantic_cache()
        if not cache:
            return SemanticCacheSearchResponse(found=False)
        
        result = cache.search(request.query_embedding, request.file_id)
        
        if result:
            return SemanticCacheSearchResponse(
                found=True,
                response=result["response"],
                similarity=result["similarity"],
                query=result.get("query", ""),
                file_id=result.get("file_id")
            )
        else:
            return SemanticCacheSearchResponse(found=False)
            
    except Exception as e:
        logger.error(f"Semantic cache search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic cache search failed: {str(e)}")


@app.post("/semantic-cache/store", response_model=SemanticCacheStoreResponse)
async def semantic_cache_store(request: SemanticCacheStoreRequest):
    """
    Store query-response pair in semantic cache
    
    Args:
        request: JSON body with query_embedding, response, query_text, and optional file_id
    
    Returns:
        Success status
    """
    try:
        cache = get_semantic_cache()
        if not cache:
            return SemanticCacheStoreResponse(
                success=False,
                message="Semantic cache is disabled"
            )
        
        success = cache.store(
            query_embedding=request.query_embedding,
            response=request.response,
            query_text=request.query_text,
            file_id=request.file_id
        )
        
        if success:
            return SemanticCacheStoreResponse(
                success=True,
                message="Stored in semantic cache"
            )
        else:
            return SemanticCacheStoreResponse(
                success=False,
                message="Failed to store in semantic cache"
            )
            
    except Exception as e:
        logger.error(f"Semantic cache store failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic cache store failed: {str(e)}")


@app.post("/semantic-cache/clear")
async def semantic_cache_clear():
    """
    Clear all entries from semantic cache
    
    Returns:
        Success message
    """
    try:
        cache = get_semantic_cache()
        if not cache:
            return {
                "success": False,
                "message": "Semantic cache is disabled"
            }
        
        cache.clear()
        return {
            "success": True,
            "message": "Semantic cache cleared"
        }
    except Exception as e:
        logger.error(f"Clear semantic cache failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear semantic cache failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

