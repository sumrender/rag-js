"""
Service for handling file ingestion, text chunking, and vector embedding.
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

import fitz
import httpx
from PIL import Image
from pymongo import MongoClient

from ..core.config import settings
from .embedding_service import EmbeddingService
from .vector_store import FAISSStore
from .cache_manager import RedisCacheManager, SemanticCache
from .file_processing_service import FileProcessingService

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Manages the file ingestion pipeline, including downloading, text extraction,
    chunking, embedding, and storage.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        text_store: FAISSStore,
        image_store: FAISSStore,
        mongo_client: MongoClient,
        cache_manager: RedisCacheManager,
        semantic_cache: Optional[SemanticCache],
        file_processing_service: FileProcessingService,
    ):
        self.embedding_service = embedding_service
        self.text_store = text_store
        self.image_store = image_store
        self.mongo_db = mongo_client[settings.MONGO_DB_NAME]
        self.cache_manager = cache_manager
        self.semantic_cache = semantic_cache
        self.file_processing_service = file_processing_service

    async def ingest_file(self, file_id: str, file_url: str, file_type: Optional[str] = None):
        """Background job for file ingestion"""
        warnings = []
        try:
            logger.info(f"Starting ingestion for file_id={file_id}, file_url={file_url}")
            self._update_mongo_status(file_id, "downloading")

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(file_url)
                response.raise_for_status()
                file_content = response.content

            file_type = file_type or self._detect_file_type(file_url)
            logger.info(f"File type detected: {file_type}")

            image_count = 0
            if file_type == 'pdf':
                image_count = await self._process_pdf(file_id, file_content, warnings)
            else:
                await self._process_text(file_id, file_content, warnings)

            self._update_mongo_status(file_id, "complete", image_count=image_count)
            self._invalidate_caches(file_id)

            logger.info(f"Ingestion complete for file_id={file_id}")
            if warnings:
                logger.warning(f"Warnings during ingestion: {warnings}")

        except Exception as e:
            error_msg = f"Ingestion failed for file_id={file_id}: {e}"
            logger.error(error_msg, exc_info=True)
            self._update_mongo_status(file_id, "error", last_error=str(e))

    async def _process_pdf(self, file_id: str, file_content: bytes, warnings: List[str]) -> int:
        """Process a PDF file for ingestion."""
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        pages = [{"text": page.get_text(), "pageNum": i + 1} for i, page in enumerate(pdf_doc)]
        full_text = "".join(p["text"] for p in pages)

        text_chunks = self._chunk_with_pages(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        await self._embed_and_store_text(file_id, text_chunks)

        extracted_images = self._extract_images_from_pdf_sync(pdf_doc, file_id, warnings)
        image_count = 0
        if extracted_images:
            image_count = await self._embed_and_store_images(file_id, extracted_images, pages)
        pdf_doc.close()
        return image_count

    async def _process_text(self, file_id: str, file_content: bytes, warnings: List[str]):
        """Process a text file for ingestion."""
        text = file_content.decode('utf-8', errors='ignore')
        text_chunks = self._chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        await self._embed_and_store_text(file_id, text_chunks)

    async def _embed_and_store_text(self, file_id: str, text_chunks: List[Dict[str, Any]]):
        """Embed text chunks and store them in the vector store."""
        if not text_chunks:
            return
        self._update_mongo_status(file_id, "embedding_text")
        chunk_texts = [chunk["text"] for chunk in text_chunks]
        text_embeddings = self.embedding_service.text_model.encode(
            chunk_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        chunk_ids = [f"{file_id}-chunk-{i}" for i in range(len(text_chunks))]
        metadatas = self._create_chunk_metadatas(file_id, text_chunks)

        self.text_store.upsert(
            ids=chunk_ids,
            embeddings=text_embeddings.tolist(),
            documents=[chunk["text"] for chunk in text_chunks],
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(text_chunks)} text chunks to FAISS")

    def _extract_images_from_pdf_sync(self, pdf_doc, file_id: str, warnings: List[str]) -> List[Dict[str, Any]]:
        """Synchronously extract images from a PDF document."""
        self._update_mongo_status(file_id, "extracting_images")
        extracted_images = []
        for page_num, page in enumerate(pdf_doc):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = pdf_doc.extract_image(xref)
                    pil_image = Image.open(io.BytesIO(base_image["image"]))
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")

                    image_id = self.file_processing_service.generate_image_id(file_id, page_num, xref, img_idx)
                    image_filename = f"{image_id}.png"
                    image_path = self.file_processing_service.images_dir / image_filename
                    pil_image.save(image_path, format="PNG")

                    bbox = None
                    try:
                        image_rects = page.get_image_rects(xref)
                        if image_rects:
                            rect = image_rects[0]
                            bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                    except Exception:
                        pass

                    extracted_images.append({
                        "image_id": image_id, "page_num": page_num, "bbox": bbox,
                        "image_path": f"images/{image_filename}", "image_url": f"/images/{image_filename}"
                    })
                except Exception as e:
                    warning_msg = f"Failed to extract image {img_idx} from page {page_num}: {e}"
                    logger.warning(warning_msg)
                    warnings.append(f"PARTIAL_IMAGE_INGESTION: {warning_msg}")
        return extracted_images

    async def _embed_and_store_images(self, file_id: str, images: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> int:
        """Embed images and store them in the vector store."""
        if not images:
            return 0
        self._update_mongo_status(file_id, "embedding_images")

        pil_images, valid_indices = [], []
        for i, img in enumerate(images):
            img_path = self.file_processing_service.images_dir / f"{img['image_id']}.png"
            if img_path.exists():
                pil_image = Image.open(img_path)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_images.append(pil_image)
                valid_indices.append(i)

        if pil_images:
            image_embeddings = self.embedding_service.clip_model.encode(
                pil_images, convert_to_numpy=True, normalize_embeddings=True, batch_size=len(pil_images), show_progress_bar=False
            )
            image_ids = [images[i]["image_id"] for i in valid_indices]
            metadatas = self._create_image_metadatas(file_id, [images[i] for i in valid_indices], pages)

            self.image_store.upsert(
                ids=image_ids,
                embeddings=image_embeddings.tolist(),
                documents=[img["image_id"] for img in [images[i] for i in valid_indices]],
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(image_ids)} image embeddings to FAISS")
            return len(image_ids)
        return 0

    def _update_mongo_status(self, file_id: str, stage: str, **kwargs):
        """Update the ingestion status in MongoDB."""
        update_doc = {"$set": {"ingestionStage": stage, "readyForChatting": stage == "complete"}}
        if kwargs:
            update_doc["$set"].update(kwargs)
        self.mongo_db.filemetadatas.update_one({"id": file_id}, update_doc, upsert=True)

    def _invalidate_caches(self, file_id: str):
        """Invalidate Redis and semantic caches for a given file."""
        if self.cache_manager:
            deleted_count = self.cache_manager.invalidate_file_cache(file_id)
            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} Redis cache entries for file_id={file_id}")
        if self.semantic_cache:
            semantic_removed = self.semantic_cache.invalidate_file(file_id)
            if semantic_removed > 0:
                logger.info(f"Invalidated {semantic_removed} semantic cache entries for file_id={file_id}")

    def _detect_file_type(self, file_url: str) -> str:
        """Detect file type from URL."""
        if file_url.lower().endswith('.pdf'):
            return 'pdf'
        return 'txt'

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Chunk text with overlap."""
        chunks, start = [], 0
        safe_overlap = min(chunk_overlap, chunk_size - 1)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "startChar": start, "endChar": end})
            next_start = end - safe_overlap
            start = end if next_start <= start else next_start
        return chunks

    def _chunk_with_pages(self, pages: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Chunk text with page tracking for PDFs."""
        full_text = "".join(p["text"] for p in pages)
        char_to_page = []
        current_char = 0
        for page in pages:
            page_length = len(page["text"])
            char_to_page.extend([{"char": current_char + i, "pageNum": page["pageNum"]} for i in range(page_length)])
            current_char += page_length

        chunks = []
        start = 0
        safe_overlap = min(chunk_overlap, chunk_size - 1)
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()
            if chunk_text:
                start_page = char_to_page[min(start, len(char_to_page) - 1)]["pageNum"] if char_to_page else 1
                end_page = char_to_page[min(end - 1, len(char_to_page) - 1)]["pageNum"] if char_to_page else 1

                chunk_data = {"text": chunk_text, "startChar": start, "endChar": end}
                if start_page == end_page:
                    chunk_data["pageNumber"] = str(start_page)
                else:
                    chunk_data["pageRange"] = f"{start_page}-{end_page}"
                chunks.append(chunk_data)

            next_start = end - safe_overlap
            start = end if next_start <= start else next_start
        return chunks

    def _create_chunk_metadatas(self, file_id: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create metadata for text chunks."""
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "fileId": file_id, "contentType": "chunk", "chunkIndex": i, "totalChunks": len(chunks),
                "characterRange": f"{chunk['startChar']}-{chunk['endChar']}"
            }
            if "pageNumber" in chunk:
                metadata["pageNumber"] = chunk["pageNumber"]
            if "pageRange" in chunk:
                metadata["pageRange"] = chunk["pageRange"]
            metadatas.append(metadata)
        return metadatas

    def _create_image_metadatas(self, file_id: str, images: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create metadata for images."""
        metadatas = []
        for img in images:
            metadata = {
                "fileId": file_id, "pageNumber": img["page_num"] + 1, "imageId": img["image_id"], "imageUrl": img["image_url"]
            }
            if img["bbox"]:
                metadata["bbox"] = json.dumps(img["bbox"])

            page_text = next((p["text"] for p in pages if p["pageNum"] == img["page_num"] + 1), "")
            metadata["nearbyText"] = page_text[:500]
            metadatas.append(metadata)
        return metadatas
