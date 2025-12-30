export interface OllamaConfig {
  baseUrl: string;
  embeddingModel: string;
  llmModel: string;
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

export interface MultimodalRetrievalConfig {
  text: {
    topK: number;
    maxK: number;
    maxChunkTokens: number;
  };
  image: {
    semantic: {
      topN: number;
      includeScores: boolean;
      minScore?: number;  // Minimum CLIP score threshold (0-1)
    };
    pageLinked: {
      maxPerPage: number;
      expandAdjacent: boolean;
      requireSemanticMatch?: boolean;  // Require semantic validation for page-linked images
    };
    maxFinal: number;
    minFusionScore?: number;  // Minimum fusion score threshold (0-1)
    maxTokens: number;
  };
  fusion: {
    weights: {
      clip: number;
      page: number;
      frequency: number;
    };
    trackProvenance: boolean;
  };
  context: {
    maxTotalTokens: number;
    systemPromptReserve: number;
  };
  queryAnalysis: {
    enabled: boolean;
    adjustWeights: boolean;
  };
}

export interface AppConfig {
  ollama: OllamaConfig;
  chunking: ChunkingConfig;
  retrieval: RetrievalConfig;
  multimodalRetrieval: MultimodalRetrievalConfig;
}

export interface AzureConfig {
  connectionString: string;
  containerName: string;
}

export interface PythonServiceConfig {
  url: string;
}