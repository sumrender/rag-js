import { OllamaConfig, ChromaConfig, ChunkingConfig, RetrievalConfig, AzureConfig } from "../models/index.js";

export class Config {
  public readonly ollama: OllamaConfig;
  public readonly chroma: ChromaConfig;
  public readonly chunking: ChunkingConfig;
  public readonly retrieval: RetrievalConfig;
  public readonly azure: AzureConfig;

  constructor() {
    this.ollama = {
      baseUrl: process.env.OLLAMA_URL ?? "http://localhost:11434",
      embeddingModel: "mxbai-embed-large",
      llmModel: "gemma3"
    };
    this.chroma = {
      url: process.env.CHROMA_URL ?? "http://localhost:8000",
      collectionName: "rag_docs"
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
  }
}

