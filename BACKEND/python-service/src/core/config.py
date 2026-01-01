
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Model settings
    CLIP_MODEL_NAME: str = "sentence-transformers/clip-ViT-B-32"
    TEXT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "cpu"

    # FAISS store settings
    FAISS_TEXT_COLLECTION: str = "text_collection"
    FAISS_IMAGE_COLLECTION: str = "images_collection"
    FAISS_DATA_DIR: str = "./faiss-data"

    # MongoDB settings
    MONGO_URI: str
    MONGO_DB_NAME: str = "simple-rag"

    # Azure Storage settings
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str = "files"

    # Cache settings
    ENABLE_CACHE: bool = False
    EMBEDDING_CACHE_SIZE: int = 1000
    CACHE_TTL_QUERY_RESULTS: int = 3600  # 1 hour
    CACHE_TTL_FAISS_RESULTS: int = 7200 # 2 hours
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    # Semantic cache settings
    SEMANTIC_CACHE_MAX_SIZE: int = 1000
    SEMANTIC_CACHE_THRESHOLD: float = 0.95

    # Text chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Server settings
    PORT: int = 8001
    PY_SERVICE_URL: str = "http://localhost:8001"

    # Path settings
    IMAGES_DIR: str = "images"

    class Config:
        """
        Pydantic settings configuration.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate settings
settings = Settings()
