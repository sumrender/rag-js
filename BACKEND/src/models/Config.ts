export interface OllamaConfig {
  baseUrl: string;
  embeddingModel: string;
  llmModel: string;
}

export interface ChromaConfig {
  url: string;
  collectionName: string;
}

export interface ChunkingConfig {
  chunkSize: number;
  chunkOverlap: number;
}

export interface RetrievalConfig {
  nResults: number;
  summaryQueryKeywords?: string[];
  pageQueryPatterns?: RegExp[];
  sourceQueryKeywords?: string[];
  includeCitations?: boolean;
  nResultsForPageQueries?: number;
  nResultsForSummaryQueries?: number;
}

export interface AppConfig {
  ollama: OllamaConfig;
  chroma: ChromaConfig;
  chunking: ChunkingConfig;
  retrieval: RetrievalConfig;
}

export interface AzureConfig {
  connectionString: string;
  containerName: string;
}