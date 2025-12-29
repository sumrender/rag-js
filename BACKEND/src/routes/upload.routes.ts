import { Router, Request, Response } from "express";
import { ServiceContainer } from "../container/ServiceContainer.js";
import { uploadTxt } from "../middleware/upload.js";
import { validateUploadRequest } from "../utils/uploadHelpers.js";

export function createUploadRoutes(container: ServiceContainer): Router {
  const router = Router();

  router.post(
    "/file-upload",
    uploadTxt.single("file"),
    async (req: Request, res: Response): Promise<void> => {
      const validation = validateUploadRequest(req, res, {
        ingestService: true,
        blobStorageService: true,
      });

      if (!validation) {
        return;
      }

      const { file, blobName, safeOriginalName } = validation;

      try {
        const uploadedBlobName = await container.blobStorageService.uploadRaw(
          file.buffer,
          blobName
        );

        container.ingestService.ingest(uploadedBlobName).catch((error: unknown) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error("Ingestion error:", errorMessage);
        });

        res.json({
          message: "File uploaded successfully. Wait a while for the ingestion to complete.",
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
        ingestService: true,
        blobStorageService: true,
      });

      if (!validation) {
        return;
      }

      const { file, blobName } = validation;

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

        // Start ingestion with progress callback
        container.ingestService.ingest(uploadedBlobName, (progressInfo) => {
          res.write(`data: ${JSON.stringify(progressInfo)}\n\n`);

          if (progressInfo.stage === 'complete') {
            res.end();
          }
        }).catch((error: unknown) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error("Ingestion error:", errorMessage);

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

