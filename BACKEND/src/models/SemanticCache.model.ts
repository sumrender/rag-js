export interface SemanticCacheSearchRequest {
  query_embedding: number[];
  file_id?: string;
}

export interface SemanticCacheSearchResponse {
  found: boolean;
  response?: string;
  similarity?: number;
  query?: string;
  file_id?: string;
}

export interface SemanticCacheStoreRequest {
  query_embedding: number[];
  response: string;
  query_text?: string;
  file_id?: string;
}

export interface SemanticCacheStoreResponse {
  success: boolean;
  message: string;
}

export interface CachedResponse {
  response: string;
  similarity: number;
  query?: string;
  file_id?: string;
}

