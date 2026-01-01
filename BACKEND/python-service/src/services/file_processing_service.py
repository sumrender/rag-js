"""
Service for processing files, including PDF parsing and image extraction.
"""
import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image
from fastapi import UploadFile

from ..core.config import settings
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class FileProcessingService:
    """
    Handles all operations related to file processing, such as
    extracting images from PDFs and generating embeddings.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.images_dir = Path(settings.IMAGES_DIR)
        self.images_dir.mkdir(exist_ok=True)

    def generate_image_id(self, file_id: str, page_num: int, xref: int, idx: int) -> str:
        """Generate stable unique image ID"""
        content = f"{file_id}|{page_num}|{xref}|{idx}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_image_id_from_url(self, image_url: str) -> str:
        """Extract image_id from image URL"""
        return image_url.replace("/images/", "").replace(".png", "")

    async def extract_images_from_pdf(
        self, file: UploadFile, file_id: Optional[str] = None, max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract images from a PDF file"""
        warnings = []
        images_result = []
        file_id = file_id or file.filename or "unknown"

        try:
            pdf_content = await file.read()
            if not pdf_content:
                warnings.append("IMAGE_EXTRACTION_FAILED: Empty PDF file")
                return {"file_id": file_id, "images": [], "warnings": warnings}

            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_images = 0

            for page_num, page in enumerate(pdf_doc):
                if max_images and total_images >= max_images:
                    warnings.append(f"Reached max_images limit: {max_images}")
                    break

                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    if max_images and total_images >= max_images:
                        break
                    try:
                        xref = img_info[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        image_id = self.generate_image_id(file_id, page_num, xref, img_idx)
                        image_filename = f"{image_id}.png"
                        image_path = self.images_dir / image_filename
                        pil_image.save(image_path, format="PNG")

                        bbox = None
                        try:
                            image_rects = page.get_image_rects(xref)
                            if image_rects:
                                rect = image_rects[0]
                                bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                        except Exception:
                            pass

                        images_result.append({
                            "image_id": image_id, "page_num": page_num, "bbox": bbox,
                            "image_path": f"images/{image_filename}", "image_url": f"/images/{image_filename}"
                        })
                        total_images += 1
                    except Exception as e:
                        warning_msg = f"Failed to extract image {img_idx} from page {page_num}: {e}"
                        logger.warning(warning_msg)
                        warnings.append(f"PARTIAL_IMAGE_INGESTION: {warning_msg}")

            pdf_doc.close()
            if not images_result and not warnings:
                warnings.append("IMAGE_EXTRACTION_FAILED: No images found in PDF")

            return {"file_id": file_id, "images": images_result, "warnings": warnings}
        except Exception as e:
            error_msg = f"IMAGE_EXTRACTION_FAILED: {e}"
            logger.error(error_msg)
            warnings.append(error_msg)
            return {"file_id": file_id, "images": [], "warnings": warnings}

    def embed_images_batch(self, image_urls: List[str], normalize: bool) -> Dict[str, Any]:
        """Generate CLIP embeddings for a batch of images"""
        warnings = []
        embeddings_result = []

        if not image_urls:
            return {"embeddings": [], "warnings": []}

        try:
            pil_images, valid_image_ids = [], []
            for image_url in image_urls:
                try:
                    image_id = self.extract_image_id_from_url(image_url)
                    image_path = self.images_dir / f"{image_id}.png"
                    if not image_path.exists():
                        warnings.append(f"IMAGE_EMBEDDING_FAILED: Image file not found for image_url: {image_url}")
                        continue

                    pil_image = Image.open(image_path)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")

                    pil_images.append(pil_image)
                    valid_image_ids.append(image_id)
                except Exception as e:
                    warning_msg = f"IMAGE_EMBEDDING_FAILED: Failed to process image {image_url}: {e}"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)

            if pil_images:
                try:
                    embeddings = self.embedding_service.clip_model.encode(
                        pil_images, convert_to_numpy=True, normalize_embeddings=normalize,
                        batch_size=len(pil_images), show_progress_bar=False
                    )
                    for image_id, embedding in zip(valid_image_ids, embeddings):
                        embeddings_result.append({"image_id": image_id, "embedding": embedding.tolist()})
                except Exception as e:
                    error_msg = f"PARTIAL_IMAGE_INGESTION: Batch embedding failed: {e}"
                    logger.error(error_msg)
                    warnings.append(error_msg)
            else:
                warnings.append("PARTIAL_IMAGE_INGESTION: No valid images to embed")

            return {"embeddings": embeddings_result, "warnings": warnings}
        except Exception as e:
            error_msg = f"IMAGE_EMBEDDING_FAILED: Unexpected error: {e}"
            logger.error(error_msg)
            return {"embeddings": [], "warnings": [error_msg]}
