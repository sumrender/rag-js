export interface FileMetadata {
  id: string;
  name: string;
  type: string;
  createdOn: string;
  path: string;
  file_url?: string;
  readyForChatting?: boolean;
  ingestionStage?: string;
  lastError?: string;
  imageCount?: number;
}
