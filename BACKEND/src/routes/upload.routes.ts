import { Router, Request, Response } from "express";
import { ServiceContainer } from "../container/ServiceContainer.js";
import { uploadTxt } from "../middleware/upload.js";
import { validateUploadRequest } from "../utils/uploadHelpers.js";
import crypto from "crypto";

export function createUploadRoutes(container: ServiceContainer): Router {
  const router = Router();

  router.post(
    "/file-upload",
    uploadTxt.single("file"),
    async (req: Request, res: Response): Promise<void> => {
      const validation = validateUploadRequest(req, res, {
        blobStorageService: true,
      });

      if (!validation) {
        return;
      }

      const { file, blobName, safeOriginalName } = validation;

      try {
        // Upload file to blob storage
        const uploadedBlobName = await container.blobStorageService.uploadRaw(
          file.buffer,
          blobName
        );

        // Get file URL for Python service
        const fileUrl = await container.blobStorageService.getFileUrl(uploadedBlobName);

        // Determine file type
        const fileExtension = safeOriginalName.split('.').pop()?.toLowerCase();
        const fileType = fileExtension === 'pdf' ? 'pdf' : 'txt';

        // Create file metadata
        const fileId = crypto.randomUUID();
        const fileMetadata = {
          id: fileId,
          name: safeOriginalName,
          type: fileType,
          createdOn: new Date().toISOString(),
          path: uploadedBlobName,
          file_url: fileUrl,
          readyForChatting: false
        };

        await container.fileRepository.save(fileMetadata);

        // Trigger Python ingestion (non-blocking)
        container.pythonServiceClient.ingestFile(fileId, fileUrl, fileType).catch((error: unknown) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error("Python ingestion error:", errorMessage);
        });

        res.json({
          message: "File uploaded successfully. Ingestion in progress.",
          fileId: fileId,
          fileName: safeOriginalName,
          blobName: uploadedBlobName,
          size: file.size,
        });
      } catch (error) {
        console.error("Upload failed:", error);
        res.status(500).json({ error: "Failed to upload file" });
      }
    }
  );

  // SSE endpoint for real-time progress updates
  router.post(
    "/upload-progress",
    uploadTxt.single("file"),
    async (req: Request, res: Response): Promise<void> => {
      const validation = validateUploadRequest(req, res, {
        blobStorageService: true,
      });

      if (!validation) {
        return;
      }

      const { file, blobName, safeOriginalName } = validation;

      // Set SSE headers
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      // Send initial connection event
      res.write(`data: ${JSON.stringify({ stage: 'start', message: 'Starting upload...', percentage: 0 })}\n\n`);

      try {
        res.write(`data: ${JSON.stringify({ stage: 'uploading', message: 'Uploading to storage...', percentage: 5 })}\n\n`);
        const uploadedBlobName = await container.blobStorageService.uploadRaw(
          file.buffer,
          blobName
        );
        res.write(`data: ${JSON.stringify({ stage: 'uploaded', message: 'Upload complete. Starting ingestion...', percentage: 10, blobName: uploadedBlobName })}\n\n`);

        // Get file URL
        const fileUrl = await container.blobStorageService.getFileUrl(uploadedBlobName);
        
        // Determine file type
        const fileExtension = safeOriginalName.split('.').pop()?.toLowerCase();
        const fileType = fileExtension === 'pdf' ? 'pdf' : 'txt';

        // Create file metadata
        const fileId = crypto.randomUUID();
        const fileMetadata = {
          id: fileId,
          name: safeOriginalName,
          type: fileType,
          createdOn: new Date().toISOString(),
          path: uploadedBlobName,
          file_url: fileUrl,
          readyForChatting: false
        };

        await container.fileRepository.save(fileMetadata);
        res.write(`data: ${JSON.stringify({ stage: 'metadata_created', message: 'File metadata created', percentage: 15, fileId: fileId })}\n\n`);

        // Trigger Python ingestion and poll status
        container.pythonServiceClient.ingestFile(fileId, fileUrl, fileType).then(() => {
          // Poll ingestion status
          const pollInterval = setInterval(async () => {
            try {
              const status = await container.pythonServiceClient.getIngestionStatus(fileId);
              res.write(`data: ${JSON.stringify({ 
                stage: status.stage || 'processing', 
                message: `Ingestion ${status.stage || 'in progress'}...`, 
                percentage: status.readyForChatting ? 100 : 50,
                readyForChatting: status.readyForChatting
              })}\n\n`);

              if (status.readyForChatting || status.error) {
                clearInterval(pollInterval);
                res.write(`data: ${JSON.stringify({ 
                  stage: status.readyForChatting ? 'complete' : 'error',
                  message: status.readyForChatting ? 'Ingestion complete' : `Ingestion failed: ${status.error}`,
                  percentage: status.readyForChatting ? 100 : 0
                })}\n\n`);
                res.end();
              }
            } catch (error) {
              clearInterval(pollInterval);
              const errorMessage = error instanceof Error ? error.message : String(error);
              res.write(`data: ${JSON.stringify({ stage: 'error', message: errorMessage, percentage: 0 })}\n\n`);
              res.end();
            }
          }, 2000); // Poll every 2 seconds

          // Timeout after 5 minutes
          setTimeout(() => {
            clearInterval(pollInterval);
            res.write(`data: ${JSON.stringify({ stage: 'timeout', message: 'Ingestion timeout', percentage: 0 })}\n\n`);
            res.end();
          }, 300000);
        }).catch((error: unknown) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error("Python ingestion error:", errorMessage);
          res.write(`data: ${JSON.stringify({ 
            stage: 'error', 
            message: errorMessage,
            percentage: 0
          })}\n\n`);
          res.end();
        });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error("Upload failed:", errorMessage);
        res.write(`data: ${JSON.stringify({ stage: 'error', message: errorMessage, percentage: 0 })}\n\n`);
        res.end();
      }
    }
  );

  return router;
}

