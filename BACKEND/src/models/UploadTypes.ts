export interface UploadValidationResult {
  file: Express.Multer.File;
  blobName: string;
  safeOriginalName: string;
}

export interface ServiceAvailability {
  blobStorageService: boolean;
}

