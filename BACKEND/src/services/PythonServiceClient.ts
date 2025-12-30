import { Config } from "../config/Config.js";

export interface IngestFileResponse {
  job_id: string;
  message: string;
}

export interface IngestionStatusResponse {
  readyForChatting: boolean;
  stage?: string;
  error?: string;
}

export interface TextChunk {
  id: string | null;
  text: string;
  metadata: Record<string, any>;
  score: number;
}

export interface RetrieveTextResponse {
  chunks: TextChunk[];
  queryModel: string;
}

export interface ImageResult {
  imageId: string;
  paintingId?: string;
  fileId?: string;
  pageNumber?: number;
  imageUrl?: string;
  score: number;
  nearbyText?: string;
}

export interface RetrieveImagesByTextResponse {
  images: ImageResult[];
}

export class PythonServiceClient {
  private baseUrl: string;

  constructor(config: Config) {
    this.baseUrl = config.pythonService.url;
  }

  async ingestFile(fileId: string, fileUrl: string, fileType?: 'pdf' | 'txt'): Promise<IngestFileResponse> {
    const response = await fetch(`${this.baseUrl}/ingest-file`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id: fileId,
        file_url: fileUrl,
        file_type: fileType
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python service ingestion failed: ${response.status} ${errorText}`);
    }

    return await response.json() as IngestFileResponse;
  }

  async getIngestionStatus(fileId: string): Promise<IngestionStatusResponse> {
    const response = await fetch(`${this.baseUrl}/ingestion-status/${fileId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return { readyForChatting: false, stage: 'not_found' };
      }
      const errorText = await response.text();
      throw new Error(`Python service status check failed: ${response.status} ${errorText}`);
    }

    return await response.json() as IngestionStatusResponse;
  }

  async retrieveText(question: string, fileId?: string, k: number = 3): Promise<RetrieveTextResponse> {
    const response = await fetch(`${this.baseUrl}/retrieve-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        file_id: fileId,
        k
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python service text retrieval failed: ${response.status} ${errorText}`);
    }

    return await response.json() as RetrieveTextResponse;
  }

  async retrieveImagesByText(question: string, fileId?: string, k: number = 3): Promise<RetrieveImagesByTextResponse> {
    const response = await fetch(`${this.baseUrl}/retrieve-images-by-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        file_id: fileId,
        k
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python service image retrieval failed: ${response.status} ${errorText}`);
    }

    return await response.json() as RetrieveImagesByTextResponse;
  }

  async retrieveImagesByPages(
    pageNumbers: number[], 
    fileId?: string, 
    maxPerPage: number = 3
  ): Promise<RetrieveImagesByTextResponse> {
    const response = await fetch(`${this.baseUrl}/retrieve-images-by-pages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        page_numbers: pageNumbers,
        file_id: fileId,
        max_per_page: maxPerPage
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python service page-linked image retrieval failed: ${response.status} ${errorText}`);
    }

    return await response.json() as RetrieveImagesByTextResponse;
  }

  async embedText(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/embed-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Python service text embedding failed: ${response.status} ${errorText}`);
    }

    const result = await response.json() as { embedding: number[] };
    return result.embedding;
  }
}

