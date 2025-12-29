import { Request, Response } from "express";
import { UploadValidationResult, ServiceAvailability } from "../models/index.js";

/**
 * Validates upload request and checks service availability.
 * Returns validation result or sends error response and returns null.
 */
export function validateUploadRequest(
  req: Request,
  res: Response,
  serviceAvailability: ServiceAvailability
): UploadValidationResult | null {
  if (!req.file) {
    res.status(400).json({ error: "TXT or PDF file is required" });
    return null;
  }

  if (!serviceAvailability.ingestService) {
    res.status(503).json({ error: "ingestService not ready yet" });
    return null;
  }
  if (!serviceAvailability.blobStorageService) {
    res.status(503).json({ error: "blobStorageService not ready yet" });
    return null;
  }

  const safeOriginalName = generateBlobName(req.file.originalname);
  const blobName = `${Date.now()}-${safeOriginalName}`;

  return {
    file: req.file,
    blobName,
    safeOriginalName,
  };
}

/**
 * Generates a safe blob name from the original filename.
 * Replaces path separators with underscores to prevent directory traversal.
 */
export function generateBlobName(originalName: string | undefined): string {
  const name = originalName || "upload";
  return name.replaceAll("/", "_").replaceAll("\\", "_");
}

