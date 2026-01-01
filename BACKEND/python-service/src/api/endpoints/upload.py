
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ..services.blob_storage_service import BlobStorageService
from ..services.ingestion_service import IngestionService
from .dependencies import get_blob_storage_service, get_ingestion_service

router = APIRouter()

def generate_blob_name(original_name: str | None) -> str:
    """Generates a safe blob name from the original filename."""
    name = original_name or "upload"
    return name.replace("/", "_").replace("\\", "_")

@router.post("/file-upload")
async def file_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = Depends(),
    blob_storage_service: BlobStorageService = Depends(get_blob_storage_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Upload a file and trigger the ingestion process."""
    if not file:
        raise HTTPException(status_code=400, detail="TXT or PDF file is required")

    safe_original_name = generate_blob_name(file.filename)
    blob_name = f"{datetime.now().timestamp()}-{safe_original_name}"

    try:
        file_buffer = await file.read()
        uploaded_blob_name = await blob_storage_service.upload_raw(file_buffer, blob_name)
        file_url = await blob_storage_service.get_file_url(uploaded_blob_name)
        file_extension = safe_original_name.split('.')[-1].lower()
        file_type = 'pdf' if file_extension == 'pdf' else 'txt'
        file_id = str(uuid.uuid4())

        ingestion_service.create_file_metadata(
            file_id, safe_original_name, file_type, datetime.now().isoformat(), uploaded_blob_name, file_url
        )

        background_tasks.add_task(
            ingestion_service.ingest_file,
            file_id,
            file_url,
            file_type,
        )

        return JSONResponse({
            "message": "File uploaded successfully. Ingestion in progress.",
            "fileId": file_id,
            "fileName": safe_original_name,
            "blobName": uploaded_blob_name,
            "size": len(file_buffer),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

@router.post("/upload-progress")
async def upload_progress(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = Depends(),
    blob_storage_service: BlobStorageService = Depends(get_blob_storage_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Upload a file and stream the ingestion progress."""
    if not file:
        raise HTTPException(status_code=400, detail="TXT or PDF file is required")

    safe_original_name = generate_blob_name(file.filename)
    blob_name = f"{datetime.now().timestamp()}-{safe_original_name}"

    async def progress_stream():
        yield f"data: {json.dumps({'stage': 'start', 'message': 'Starting upload...', 'percentage': 0})}\n\n"

        try:
            file_buffer = await file.read()
            yield f"data: {json.dumps({'stage': 'uploading', 'message': 'Uploading to storage...', 'percentage': 5})}\n\n"
            uploaded_blob_name = await blob_storage_service.upload_raw(file_buffer, blob_name)
            yield f"data: {json.dumps({'stage': 'uploaded', 'message': 'Upload complete. Starting ingestion...', 'percentage': 10, 'blobName': uploaded_blob_name})}\n\n"

            file_url = await blob_storage_service.get_file_url(uploaded_blob_name)
            file_extension = safe_original_name.split('.')[-1].lower()
            file_type = 'pdf' if file_extension == 'pdf' else 'txt'
            file_id = str(uuid.uuid4())

            ingestion_service.create_file_metadata(
                file_id, safe_original_name, file_type, datetime.now().isoformat(), uploaded_blob_name, file_url
            )
            yield f"data: {json.dumps({'stage': 'metadata_created', 'message': 'File metadata created', 'percentage': 15, 'fileId': file_id})}\n\n"

            background_tasks.add_task(
                ingestion_service.ingest_file,
                file_id,
                file_url,
                file_type,
            )

            async for status in ingestion_service.poll_ingestion_status(file_id):
                ready = status.get("readyForChatting", False)
                error = status.get("lastError")

                yield f"data: {json.dumps({'stage': status.get('ingestionStage', 'processing'), 'message': f'Ingestion {status.get('ingestionStage', 'in progress')}...', 'percentage': 100 if ready else 50, 'readyForChatting': ready})}\n\n"

                if ready or error:
                    yield f"data: {json.dumps({'stage': 'complete' if ready else 'error', 'message': 'Ingestion complete' if ready else f'Ingestion failed: {error}', 'percentage': 100 if ready else 0})}\n\n"
                    break
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e), 'percentage': 0})}\n\n"

    return StreamingResponse(progress_stream())
