from azure.storage.blob.aio import BlobServiceClient
from ..core.config import settings

class BlobStorageService:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME

    async def upload_raw(self, data: bytes, blob_name: str) -> str:
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
        await blob_client.upload_blob(data, overwrite=True)
        return blob_name

    async def get_file_url(self, blob_name: str) -> str:
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
        return blob_client.url
