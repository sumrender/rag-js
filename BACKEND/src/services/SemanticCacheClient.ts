import { Config } from "../config/Config.js";
import { PythonServiceClient } from "./PythonServiceClient.js";
import {
  SemanticCacheSearchRequest,
  SemanticCacheSearchResponse,
  SemanticCacheStoreRequest,
  CachedResponse
} from "../models/SemanticCache.model.js";

export class SemanticCacheClient {
  private baseUrl: string;
  private enabled: boolean;
  private threshold: number;
  private pythonServiceClient: PythonServiceClient;

  constructor(config: Config, pythonServiceClient: PythonServiceClient) {
    this.baseUrl = config.pythonService.url;
    this.enabled = config.semanticCache.enabled;
    this.threshold = config.semanticCache.similarityThreshold;
    this.pythonServiceClient = pythonServiceClient;
  }

  /**
   * Find similar cached query
   * @param question User question text
   * @param fileId Optional file ID to scope search
   * @returns Cached response if similarity > threshold, null otherwise
   */
  async findSimilar(question: string, fileId?: string): Promise<CachedResponse | null> {
    if (!this.enabled) {
      return null;
    }

    try {
      // Generate query embedding
      const queryEmbedding = await this.pythonServiceClient.embedText(question);

      // Search semantic cache
      const request: SemanticCacheSearchRequest = {
        query_embedding: queryEmbedding,
        file_id: fileId
      };

      const response = await fetch(`${this.baseUrl}/semantic-cache/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.warn(`Semantic cache search failed: ${response.status} ${errorText}`);
        return null;
      }

      const result = await response.json() as SemanticCacheSearchResponse;

      if (result.found && result.response && result.similarity !== undefined) {
        // Double-check threshold (should be done server-side, but verify)
        if (result.similarity >= this.threshold) {
          return {
            response: result.response,
            similarity: result.similarity,
            query: result.query,
            file_id: result.file_id
          };
        }
      }

      return null;
    } catch (error) {
      console.warn(`Semantic cache search error: ${error}`);
      return null;
    }
  }

  /**
   * Store query-response pair in semantic cache
   * @param question User question text
   * @param response LLM response text
   * @param fileId Optional file ID
   */
  async store(question: string, response: string, fileId?: string): Promise<void> {
    if (!this.enabled) {
      return;
    }

    try {
      // Generate query embedding
      const queryEmbedding = await this.pythonServiceClient.embedText(question);

      // Store in semantic cache
      const request: SemanticCacheStoreRequest = {
        query_embedding: queryEmbedding,
        response: response,
        query_text: question,
        file_id: fileId
      };

      const fetchResponse = await fetch(`${this.baseUrl}/semantic-cache/store`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!fetchResponse.ok) {
        const errorText = await fetchResponse.text();
        console.warn(`Semantic cache store failed: ${fetchResponse.status} ${errorText}`);
      }
    } catch (error) {
      console.warn(`Semantic cache store error: ${error}`);
      // Don't throw - caching failures shouldn't break the request
    }
  }

  /**
   * Clear semantic cache
   */
  async clear(): Promise<boolean> {
    if (!this.enabled) {
      return false;
    }

    try {
      const response = await fetch(`${this.baseUrl}/semantic-cache/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return false;
      }

      const result = await response.json() as { success: boolean; message?: string };
      return result.success === true;
    } catch (error) {
      console.warn(`Clear semantic cache error: ${error}`);
      return false;
    }
  }
}

