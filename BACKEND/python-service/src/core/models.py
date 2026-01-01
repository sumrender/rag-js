"""
Pydantic models for API requests and responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class EmbedImagesBatchRequest(BaseModel):
    """Request model for embed-images-batch endpoint"""
    file_id: str
    image_urls: List[str]
    normalize: bool = True

class EmbedTextRequest(BaseModel):
    """Request model for embed-text endpoint"""
    text: str

class EmbedTextResponse(BaseModel):
    """Response model for embed-text endpoint"""
    embedding: List[float]

class IngestFileRequest(BaseModel):
    """Request model for ingest-file endpoint"""
    file_id: str
    file_url: str
    file_type: Optional[str] = None  # 'pdf' or 'txt', auto-detected if not provided

class IngestFileResponse(BaseModel):
    """Response model for ingest-file endpoint"""
    job_id: str
    message: str

class RetrieveTextRequest(BaseModel):
    """Request model for retrieve-text endpoint"""
    question: str
    file_id: Optional[str] = None
    k: int = 3

class RetrieveTextResponse(BaseModel):
    """Response model for retrieve-text endpoint"""
    chunks: List[Dict[str, Any]]
    queryModel: str

class RetrieveImagesByTextRequest(BaseModel):
    """Request model for retrieve-images-by-text endpoint"""
    question: str
    file_id: Optional[str] = None
    k: int = 3

class RetrieveImagesByTextResponse(BaseModel):
    """Response model for retrieve-images-by-text endpoint"""
    images: List[Dict[str, Any]]

class RetrieveImagesByPagesRequest(BaseModel):
    """Request model for retrieve-images-by-pages endpoint"""
    file_id: Optional[str] = None
    page_numbers: List[int]  # 1-indexed
    max_per_page: int = 3

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
