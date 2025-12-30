import { OllamaConfig, ChunkingConfig, RetrievalConfig, AzureConfig, PythonServiceConfig, MultimodalRetrievalConfig } from "../models/index.js";

export class Config {
  public readonly ollama: OllamaConfig;
  public readonly chunking: ChunkingConfig;
  public readonly retrieval: RetrievalConfig;
  public readonly multimodalRetrieval: MultimodalRetrievalConfig;
  public readonly azure: AzureConfig;
  public readonly pythonService: PythonServiceConfig;

  constructor() {
    this.ollama = {
      baseUrl: process.env.OLLAMA_URL ?? "http://localhost:11434",
      embeddingModel: "mxbai-embed-large",
      llmModel: "gemma3"
    };
    this.chunking = {
      chunkSize: 1000,      // characters per chunk
      chunkOverlap: 200     // overlap between chunks
    };
    this.retrieval = {
      nResults: 3,           // number of chunks to retrieve
      summaryQueryKeywords: [
        "summary", "summarize", "overview", "what is", "about", 
        "main points", "key points", "what is this about", "what's this about"
      ],
      pageQueryPatterns: [
        /page\s+(\d+)/i,
        /on\s+page\s+(\d+)/i,
        /what.*page\s+(\d+)/i,
        /page\s+(\d+).*content/i,
        /content.*page\s+(\d+)/i
      ],
      sourceQueryKeywords: [
        "source", "reference", "which page", "what page", 
        "where", "citation", "cite", "from which page"
      ],
      includeCitations: true,
      nResultsForPageQueries: 10,
      nResultsForSummaryQueries: 5
    };
    this.azure = {
      connectionString: process.env.AZURE_STORAGE_CONNECTION_STRING ?? "UseDevelopmentStorage=true",
      containerName: process.env.AZURE_STORAGE_CONTAINER ?? "documents"
    };
    this.pythonService = {
      url: process.env.PYTHON_SERVICE_URL ?? "http://localhost:8001"
    };
    this.multimodalRetrieval = {
      text: {
        topK: 7,
        maxK: 10,
        maxChunkTokens: 5600,  // 70% of 8000
      },
      image: {
        semantic: {
          topN: 4,  // Reduced from 7 to get only highly relevant images
          includeScores: true,
          minScore: 0.25,  // Minimum CLIP similarity score (0-1 range) - increased for better filtering
        },
        pageLinked: {
          maxPerPage: 3,
          expandAdjacent: false,  // Future enhancement
          requireSemanticMatch: false,  // Can be enabled for stricter filtering
        },
        maxFinal: 5,  // Reduced from 10 to return fewer final results
        minFusionScore: 0.15,  // Minimum fusion score to include an image
        maxTokens: 1600,  // 20% of 8000
      },
      fusion: {
        weights: {
          clip: 0.65,  // Increased from 0.5 to prioritize semantic relevance
          page: 0.25,  // Decreased from 0.3
          frequency: 0.2,
        },
        trackProvenance: true,
      },
      context: {
        maxTotalTokens: 8000,
        systemPromptReserve: 800,  // 10% of 8000
      },
      queryAnalysis: {
        enabled: true,
        adjustWeights: true,
      },
    };
  }
}

