export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  images?: Array<{
    imageId: string;
    imageUrl: string;
    pageNumber?: number;
    score: number;
  }>;
}

export interface ChatResponse {
  ragAnswer: string;
  simpleAnswer: string;
}

export interface FileUploadResponse {
  message: string;
  fileName: string;
  filePath: string;
  size: number;
}

export interface UploadProgressEvent {
  stage: 'start' | 'uploading' | 'uploaded' | 'chunking' | 'summarizing' | 'embedding' | 'complete' | 'error';
  current?: number;
  total?: number;
  message: string;
  percentage?: number;
}

export interface FileMetadata {
  id: string;
  name: string;
  type: string;
  createdOn: string;
  path: string;
}
