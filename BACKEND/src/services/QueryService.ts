import { OllamaService } from "./OllamaService.js";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";
import { PythonServiceClient, TextChunk, ImageResult } from "./PythonServiceClient.js";
import { SemanticCacheClient } from "./SemanticCacheClient.js";

type QueryType = 'summary' | 'page' | 'source' | 'detail';

interface QueryAnalysis {
  type: QueryType;
  pageNumber?: number;
}

interface ImageResultWithProvenance extends ImageResult {
  sources: {
    semantic: boolean;
    pageLinked: boolean;
    clipRank?: number;
    pageList?: number[];
  };
  fusionScore: number;
}

export class QueryService {
  constructor(
    private ollamaService: OllamaService,
    private config: Config,
    private pythonServiceClient: PythonServiceClient,
    private semanticCacheClient: SemanticCacheClient
  ) {}

  private detectQueryType(question: string): QueryAnalysis {
    const lowerQuestion = question.toLowerCase();
    
    // Check for summary queries
    const summaryKeywords = this.config.retrieval.summaryQueryKeywords || [];
    if (summaryKeywords.some(keyword => lowerQuestion.includes(keyword.toLowerCase()))) {
      return { type: 'summary' };
    }
    
    // Check for page-specific queries
    const pagePatterns = this.config.retrieval.pageQueryPatterns || [];
    for (const pattern of pagePatterns) {
      const match = question.match(pattern);
      if (match && match[1]) {
        const pageNum = parseInt(match[1], 10);
        if (!isNaN(pageNum) && pageNum > 0) {
          return { type: 'page', pageNumber: pageNum };
        }
      }
    }
    
    // Check for source reference queries
    const sourceKeywords = this.config.retrieval.sourceQueryKeywords || [];
    if (sourceKeywords.some(keyword => lowerQuestion.includes(keyword.toLowerCase()))) {
      return { type: 'source' };
    }
    
    // Default to detail query
    return { type: 'detail' };
  }

  async ask(question: string): Promise<string> {
    // Use Python service for retrieval
    const retrievalResult = await this.pythonServiceClient.retrieveText(
      question,
      undefined,
      this.config.retrieval.nResults
    );
    
    if (!retrievalResult.chunks || retrievalResult.chunks.length === 0) {
      throw new Error(ResultCodes.NO_RELEVANT_DOCUMENTS);
    }
    
    const context = retrievalResult.chunks.map(chunk => chunk.text).join("\n\n---\n\n");
    
    const prompt = `
Use the following context to answer the question. If the context doesn't contain enough information, say so.

Context:
${context}

Question: ${question}

Answer:
`;
    
    return await this.ollamaService.generate(prompt);
  }

  async askSimple(question: string): Promise<string> {
    return await this.ollamaService.generate(question);
  }


  private formatContextWithCitations(
    documents: string[],
    metadatas: Array<Record<string, any>> | undefined
  ): string {
    if (!documents || documents.length === 0) {
      return "";
    }

    const formattedParts: string[] = [];
    
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const metadata = metadatas?.[i];
      
      let citation = "";
      if (metadata) {
        if (metadata.pageNumber) {
          citation = `[Page ${metadata.pageNumber}]`;
        } else if (metadata.pageRange) {
          citation = `[Pages ${metadata.pageRange}]`;
        }
      }
      
      if (citation) {
        formattedParts.push(`${citation}\n${doc}`);
      } else {
        formattedParts.push(doc);
      }
    }
    
    return formattedParts.join("\n\n---\n\n");
  }

  private estimateTokens(text: string): number {
    // Conservative estimate: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  private estimateImageTokens(image: ImageResultWithProvenance): number {
    const nearbyTextLen = image.nearbyText?.length || 0;
    // nearbyText + metadata overhead (imageId, pageNumber, etc.)
    return Math.ceil(nearbyTextLen / 4) + 50;
  }

  private applyContextBudget(
    chunks: TextChunk[],
    images: ImageResultWithProvenance[]
  ): { chunks: TextChunk[]; images: ImageResultWithProvenance[] } {
    const mmConfig = this.config.multimodalRetrieval;

    const textBudget = mmConfig.text.maxChunkTokens;
    const imageBudget = mmConfig.image.maxTokens;

    // Calculate current token usage
    let textTokens = 0;
    const validChunks: TextChunk[] = [];
    for (const chunk of chunks) {
      const chunkTokens = this.estimateTokens(chunk.text);
      if (textTokens + chunkTokens <= textBudget) {
        textTokens += chunkTokens;
        validChunks.push(chunk);
      } else {
        // If a single chunk exceeds budget, truncate it
        if (chunkTokens > textBudget * 0.8) {
          const maxChars = Math.floor(textBudget * 0.8 * 4);
          const truncatedChunk: TextChunk = {
            ...chunk,
            text: chunk.text.slice(0, maxChars) + "... [truncated]",
          };
          validChunks.push(truncatedChunk);
        }
        break; // Stop adding chunks if we'd exceed budget
      }
    }

    // Calculate image token usage
    let imageTokens = 0;
    const validImages: ImageResultWithProvenance[] = [];
    for (const img of images) {
      const imgTokens = this.estimateImageTokens(img);
      if (imageTokens + imgTokens <= imageBudget) {
        imageTokens += imgTokens;
        validImages.push(img);
      } else {
        break; // Stop adding images if we'd exceed budget
      }
    }

    return {
      chunks: validChunks,
      images: validImages,
    };
  }

  private async retrieveMultimodal(
    question: string,
    fileId?: string,
    queryType?: QueryType
  ): Promise<{
    textChunks: TextChunk[];
    images: ImageResultWithProvenance[];
  }> {
    const mmConfig = this.config.multimodalRetrieval;

    // Stage 1: Text Retrieval
    const textResult = await this.pythonServiceClient.retrieveText(
      question,
      fileId,
      mmConfig.text.topK
    );
    const textChunks = textResult.chunks || [];

    // Extract page numbers from text chunks
    const pageNumbers = new Set<number>();
    for (const chunk of textChunks) {
      const meta = chunk.metadata || {};
      if (meta.pageNumber) {
        const pageNum = typeof meta.pageNumber === 'string' 
          ? parseInt(meta.pageNumber, 10) 
          : meta.pageNumber;
        if (!isNaN(pageNum) && pageNum > 0) {
          pageNumbers.add(pageNum);
        }
      } else if (meta.pageRange) {
        // Parse page range like "10-15"
        const parts = meta.pageRange.split('-');
        if (parts.length === 2) {
          const start = parseInt(parts[0], 10);
          const end = parseInt(parts[1], 10);
          if (!isNaN(start) && !isNaN(end)) {
            for (let p = start; p <= end; p++) {
              pageNumbers.add(p);
            }
          }
        }
      }
    }

    // Stage 2: Semantic Image Retrieval
    let semanticImages: ImageResultWithProvenance[] = [];
    try {
      const semanticResult = await this.pythonServiceClient.retrieveImagesByText(
        question,
        fileId,
        mmConfig.image.semantic.topN
      );
      const minClipScore = mmConfig.image.semantic.minScore ?? 0;
      const allSemanticImages = semanticResult.images || [];
      const filteredSemanticImages = allSemanticImages.filter((img) => {
        // Filter by minimum CLIP score threshold
        return img.score >= minClipScore;
      });
      
      console.log(`[Image Retrieval] Semantic: ${allSemanticImages.length} retrieved, ${filteredSemanticImages.length} passed minClipScore (${minClipScore})`);
      if (filteredSemanticImages.length < allSemanticImages.length) {
        const filteredOut = allSemanticImages.filter(img => img.score < minClipScore);
        console.log(`[Image Retrieval] Filtered out ${filteredOut.length} images with scores:`, 
          filteredOut.map(img => ({ imageId: img.imageId, score: img.score.toFixed(3) })));
      }
      
      semanticImages = filteredSemanticImages.map((img, idx) => ({
          ...img,
          sources: {
            semantic: true,
            pageLinked: false,
            clipRank: idx + 1,
          },
          fusionScore: 0, // Will be calculated in fusion step
        }));
    } catch (error) {
      console.warn("Semantic image retrieval failed (non-fatal):", error);
    }

    // Stage 3: Page-Linked Image Retrieval
    let pageLinkedImages: ImageResultWithProvenance[] = [];
    if (pageNumbers.size > 0) {
      try {
        const pageLinkedResult = await this.pythonServiceClient.retrieveImagesByPages(
          Array.from(pageNumbers),
          fileId,
          mmConfig.image.pageLinked.maxPerPage
        );
        pageLinkedImages = (pageLinkedResult.images || []).map((img) => ({
          ...img,
          sources: {
            semantic: false,
            pageLinked: true,
            pageList: [img.pageNumber!].filter(p => p !== undefined),
          },
          fusionScore: 0, // Will be calculated in fusion step
        }));
      } catch (error) {
        console.warn("Page-linked image retrieval failed (non-fatal):", error);
      }
    }

    // Fusion & Deduplication
    const imageMap = new Map<string, ImageResultWithProvenance>();

    // Process semantic images
    for (const img of semanticImages) {
      const key = img.imageId;
      if (!key) continue;

      if (imageMap.has(key)) {
        const existing = imageMap.get(key)!;
        // Merge: keep max clip score, update sources
        existing.sources.semantic = true;
        if (img.sources.clipRank && (!existing.sources.clipRank || img.sources.clipRank < existing.sources.clipRank)) {
          existing.sources.clipRank = img.sources.clipRank;
        }
        if (img.score > existing.score) {
          existing.score = img.score;
        }
      } else {
        imageMap.set(key, { ...img });
      }
    }

    // Process page-linked images
    for (const img of pageLinkedImages) {
      const key = img.imageId;
      if (!key) continue;

      if (imageMap.has(key)) {
        const existing = imageMap.get(key)!;
        // Merge: mark as page-linked, add to page list
        existing.sources.pageLinked = true;
        if (img.pageNumber) {
          if (!existing.sources.pageList) {
            existing.sources.pageList = [];
          }
          if (!existing.sources.pageList.includes(img.pageNumber)) {
            existing.sources.pageList.push(img.pageNumber);
          }
        }
      } else {
        imageMap.set(key, { ...img });
      }
    }

    // Calculate fusion scores
    let weights = { ...mmConfig.fusion.weights };
    
    // Apply query-type weight adjustments if enabled
    if (mmConfig.queryAnalysis.enabled && mmConfig.queryAnalysis.adjustWeights && queryType) {
      const adjustments: Record<QueryType, { clip: number; page: number; frequency: number }> = {
        summary: { clip: -0.1, page: 0, frequency: +0.1 },
        page: { clip: 0, page: +0.05, frequency: 0 },
        source: { clip: 0, page: +0.05, frequency: 0 },
        detail: { clip: 0, page: 0, frequency: 0 },
      };
      
      const adj = adjustments[queryType] || { clip: 0, page: 0, frequency: 0 };
      weights = {
        clip: Math.max(0, Math.min(1, weights.clip + adj.clip)),
        page: Math.max(0, Math.min(1, weights.page + adj.page)),
        frequency: Math.max(0, Math.min(1, weights.frequency + adj.frequency)),
      };
    }

    // Count frequency (how many times image appears across different sources)
    const frequencyMap = new Map<string, number>();
    for (const img of imageMap.values()) {
      let freq = 0;
      if (img.sources.semantic) freq++;
      if (img.sources.pageLinked) freq++;
      frequencyMap.set(img.imageId!, freq);
    }

    // Calculate fusion scores for all images
    for (const img of imageMap.values()) {
      const clipScore = img.sources.semantic ? img.score : 0;
      const pageScore = img.sources.pageLinked ? 1.0 : 0;
      const frequencyScore = (frequencyMap.get(img.imageId!) || 0) / 2.0; // Normalize to 0-1

      img.fusionScore = 
        weights.clip * clipScore +
        weights.page * pageScore +
        weights.frequency * frequencyScore;
    }

    // Filter by minimum fusion score, then sort and take top N
    const minFusionScore = mmConfig.image.minFusionScore ?? 0;
    const minClipScore = mmConfig.image.semantic.minScore ?? 0;
    
    const allImages = Array.from(imageMap.values());
    const filteredImages = allImages.filter((img) => {
      // CRITICAL: Only include images that have semantic matches
      // Page-linked images without semantic relevance should be excluded
      if (!img.sources.semantic) {
        return false; // Exclude all images without semantic matches
      }
      
      // Ensure semantic images meet minimum CLIP score
      if (img.score < minClipScore) {
        return false;
      }
      
      // Filter by minimum fusion score threshold
      return img.fusionScore >= minFusionScore;
    });
    
    console.log(`[Image Retrieval] Final: ${allImages.length} candidates, ${filteredImages.length} passed filters (minFusionScore: ${minFusionScore}, minClipScore: ${minClipScore})`);
    if (filteredImages.length < allImages.length) {
      const filteredOut = allImages.filter(img => 
        !img.sources.semantic || img.score < minClipScore || img.fusionScore < minFusionScore
      );
      console.log(`[Image Retrieval] Filtered out ${filteredOut.length} images:`, 
        filteredOut.map(img => ({ 
          imageId: img.imageId, 
          semantic: img.sources.semantic, 
          clipScore: img.score.toFixed(3), 
          fusionScore: img.fusionScore.toFixed(3) 
        })));
    }
    
    const sortedImages = filteredImages
      .sort((a, b) => b.fusionScore - a.fusionScore)
      .slice(0, mmConfig.image.maxFinal);

    return {
      textChunks,
      images: sortedImages,
    };
  }

  async *askStream(history: { role: string; content: string }[], fileId?: string): AsyncGenerator<{type: 'text' | 'images', data: any}> {
    const lastUserMessage = history.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) {
       throw new Error("No user message found in history");
    }
    const question = lastUserMessage.content;

    // Check semantic cache first
    if (this.config.semanticCache.enabled) {
      const cached = await this.semanticCacheClient.findSimilar(question, fileId);
      if (cached && cached.similarity >= this.config.semanticCache.similarityThreshold) {
        console.log(`[Semantic Cache HIT] Similarity: ${cached.similarity.toFixed(4)}, Query: "${cached.query || question.substring(0, 50)}..."`);
        // Yield cached response as text chunks (simulate streaming)
        const responseText = cached.response;
        // Split into chunks for streaming effect
        const chunkSize = 50; // characters per chunk
        for (let i = 0; i < responseText.length; i += chunkSize) {
          yield { type: 'text', data: responseText.slice(i, i + chunkSize) };
        }
        return;
      }
    }

    const queryAnalysis = this.detectQueryType(question);
    
    // Use multimodal retrieval
    let textChunks: TextChunk[] = [];
    let images: ImageResultWithProvenance[] = [];
    
    try {
      const retrievalResult = await this.retrieveMultimodal(
        question,
        fileId,
        queryAnalysis.type
      );
      
      textChunks = retrievalResult.textChunks;
      images = retrievalResult.images;

      // Empty result handling
      if (textChunks.length === 0 && images.length === 0) {
        yield { 
          type: 'text', 
          data: "I couldn't find relevant information in the documents."
        };
        return;
      }

      // Apply context budget
      const budgeted = this.applyContextBudget(textChunks, images);
      textChunks = budgeted.chunks;
      images = budgeted.images;

      // Yield images if any were retrieved
      if (images.length > 0) {
        // Construct full URLs: python-api-url + image_path
        const pythonServiceUrl = this.config.pythonService.url;
        const imagesWithFullUrls = images.map(img => ({
          ...img,
          imageUrl: img.imageUrl 
            ? `${pythonServiceUrl}${img.imageUrl}` 
            : undefined
        }));
        yield { type: 'images', data: imagesWithFullUrls };
      }
    } catch (error) {
      console.error("Retrieval failed:", error);
      throw error;
    }
    
    // Build context from text chunks
    const retrievedMetadata = textChunks.map(c => c.metadata);
    const context = this.formatContextWithCitations(
      textChunks.map(c => c.text),
      retrievedMetadata
    );
    
    // Build system prompt with citation instructions
    let systemPrompt = `Use the following context to answer the user's question. If the context doesn't contain enough information, you can say so or use your general knowledge, but prioritize the context.
IMPORTANT => Don't use markdown or any other formatting. Always return answer in plain text.`;

    if (this.config.retrieval.includeCitations && retrievedMetadata) {
      systemPrompt += `\n\nWhen answering, if the context includes page numbers, mention them naturally in your response.
For example: "According to page 10, ..." or "As mentioned on pages 10-11, ..."
If asked about sources or references, explicitly list the page numbers you used.`;
    }

    if (queryAnalysis.type === 'source') {
      systemPrompt += `\n\nThe user is asking about sources or references. Explicitly list the page numbers and provide brief context for each source you're referencing.`;
    }

    systemPrompt += `\n\nContext:\n${context}`;
    
    // Add image context if available
    if (images.length > 0) {
      systemPrompt += `\n\nRelevant images found: ${images.length} image(s) related to the query.`;
      if (this.config.multimodalRetrieval.fusion.trackProvenance) {
        // Add provenance information for explainability
        const imageInfo = images.map((img, idx) => {
          const sources = [];
          if (img.sources.semantic) sources.push(`semantic match (rank ${img.sources.clipRank || '?'})`);
          if (img.sources.pageLinked) sources.push(`page ${img.sources.pageList?.join(', ') || '?'}`);
          return `Image ${idx + 1}: ${sources.join(' + ')}`;
        }).join('\n');
        systemPrompt += `\n\nImage sources:\n${imageInfo}`;
      }
    }
    
    // Create new messages array with system prompt at the start
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history
    ];

    // Collect full response for caching
    let fullResponse = '';

    // Yield text chunks with type
    for await (const textChunk of this.ollamaService.chatStream(messages)) {
      fullResponse += textChunk;
      yield { type: 'text', data: textChunk };
    }

    // Store in semantic cache after generation
    if (this.config.semanticCache.enabled && fullResponse.trim().length > 0) {
      // Store asynchronously (don't await to avoid blocking response)
      this.semanticCacheClient.store(question, fullResponse, fileId).catch(err => {
        console.warn(`Failed to store in semantic cache: ${err}`);
      });
    }
  }
}

