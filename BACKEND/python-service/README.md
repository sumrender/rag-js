# Python Sidecar Service

FastAPI service for PDF image extraction and CLIP embedding generation.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note for macOS**: If you encounter build errors with Pillow (JPEG library not found), you may need to install system dependencies:
```bash
brew install jpeg libpng libtiff webp
```

Alternatively, you can use Python 3.11 or 3.12 which have better pre-built wheel support.

3. Configure environment (optional):
```bash
# Create .env file with:
PY_SERVICE_URL=http://localhost:8001
CLIP_MODEL_NAME=sentence-transformers/clip-ViT-B-32
DEVICE=cpu
```

## Running the Service

### Using the start script (recommended):
```bash
./start.sh
```

### Manual start:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Or directly:
```bash
python main.py
```

## API Endpoints

### GET /health
Health check endpoint. Returns service status and model information.

### POST /extract-images
Extract images from a PDF file.

**Request (multipart/form-data):**
- `file`: PDF file (required)
- `file_id`: Optional file identifier
- `max_images`: Optional maximum number of images to extract

**Response:**
```json
{
  "file_id": "example.pdf",
  "images": [
    {
      "image_id": "abc123...",
      "page_num": 0,
      "bbox": [0, 0, 100, 100],
      "image_path": "images/abc123....png",
      "image_url": "/images/abc123....png"
    }
  ],
  "warnings": []
}
```

### POST /embed-images-batch
Generate CLIP embeddings for a batch of images (BATCH ONLY).

**Request (JSON body):**
```json
{
  "file_id": "example.pdf",
  "image_urls": [
    "/images/abc123....png",
    "/images/def456....png"
  ],
  "normalize": true
}
```
- `file_id`: File identifier (required)
- `image_urls`: Array of image URLs from `/extract-images` endpoint (required)
- `normalize`: Boolean, whether to normalize embeddings (default: true)

**Response:**
```json
{
  "embeddings": [
    {
      "image_id": "abc123...",
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "warnings": []
}
```

## Notes

- The CLIP model is loaded once at startup and reused for all requests
- All image processing errors are non-fatal and return warnings
- Empty inputs are handled gracefully
- Only batch embedding is supported (no single-image endpoint)

