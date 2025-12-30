import { OllamaService } from "./OllamaService.js";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";
import { PythonServiceClient } from "./PythonServiceClient.js";

type QueryType = 'summary' | 'page' | 'source' | 'detail';

interface QueryAnalysis {
  type: QueryType;
  pageNumber?: number;
}

export class QueryService {
  constructor(
    private ollamaService: OllamaService,
    private config: Config,
    private pythonServiceClient: PythonServiceClient
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

  private pageInRange(pageRange: string, pageNum: number): boolean {
    // Check if pageNum is within a range like "10-11" or "10-15"
    const parts = pageRange.split('-');
    if (parts.length === 2) {
      const start = parseInt(parts[0], 10);
      const end = parseInt(parts[1], 10);
      if (!isNaN(start) && !isNaN(end)) {
        return pageNum >= start && pageNum <= end;
      }
    }
    return false;
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

  async *askStream(history: { role: string; content: string }[], fileId?: string): AsyncGenerator<string> {
    const lastUserMessage = history.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) {
       throw new Error("No user message found in history");
    }
    const question = lastUserMessage.content;

    const queryAnalysis = this.detectQueryType(question);
    let context = "";
    let retrievedMetadata: Array<Record<string, any>> | undefined;
    let imageResults: any[] = [];
    
    // Use Python service for text retrieval
    try {
      let k = this.config.retrieval.nResults;
      
      if (queryAnalysis.type === 'summary') {
        // For summary queries, get more results to find summaries
        k = Math.max(this.config.retrieval.nResultsForSummaryQueries || 5, this.config.retrieval.nResults || 3);
      } else if (queryAnalysis.type === 'page' && queryAnalysis.pageNumber) {
        // For page queries, get more results to filter by page
        k = Math.max(20, (this.config.retrieval.nResultsForPageQueries || 10) * 2);
      }
      
      const retrievalResult = await this.pythonServiceClient.retrieveText(
        question,
        fileId,
        k
      );
      
      // Filter by query type
      if (queryAnalysis.type === 'summary') {
        // Prioritize summaries, then chunks
        const summaryChunks = retrievalResult.chunks.filter(c => c.metadata?.contentType === 'summary');
        const regularChunks = retrievalResult.chunks.filter(c => c.metadata?.contentType === 'chunk');
        const allChunks = [...summaryChunks, ...regularChunks].slice(0, this.config.retrieval.nResultsForSummaryQueries || 5);
        
        context = this.formatContextWithCitations(
          allChunks.map(c => c.text),
          allChunks.map(c => c.metadata)
        );
        retrievedMetadata = allChunks.map(c => c.metadata);
      } else if (queryAnalysis.type === 'page' && queryAnalysis.pageNumber) {
        // Filter by page number
        const pageNum = queryAnalysis.pageNumber;
        const pageNumStr = pageNum.toString();
        const filteredChunks = retrievalResult.chunks.filter(chunk => {
          const meta = chunk.metadata || {};
          return meta.pageNumber === pageNumStr ||
                 (meta.pageRange && meta.pageRange.includes(pageNumStr)) ||
                 (meta.pageRange && this.pageInRange(meta.pageRange, pageNum));
        });
        
        const chunksToUse = filteredChunks.length > 0 
          ? filteredChunks.slice(0, this.config.retrieval.nResultsForPageQueries || 10)
          : retrievalResult.chunks.slice(0, this.config.retrieval.nResultsForPageQueries || 10);
        
        context = this.formatContextWithCitations(
          chunksToUse.map(c => c.text),
          chunksToUse.map(c => c.metadata)
        );
        retrievedMetadata = chunksToUse.map(c => c.metadata);
      } else {
        // Standard retrieval
        const chunksToUse = retrievalResult.chunks.slice(0, this.config.retrieval.nResults);
        context = this.formatContextWithCitations(
          chunksToUse.map(c => c.text),
          chunksToUse.map(c => c.metadata)
        );
        retrievedMetadata = chunksToUse.map(c => c.metadata);
      }
      
      // Optionally retrieve images by text (heuristic: keywords like "painting", "diagram", "image")
      const imageKeywords = ['painting', 'diagram', 'image', 'picture', 'figure', 'chart', 'graph', 'show me'];
      const shouldRetrieveImages = imageKeywords.some(keyword => question.toLowerCase().includes(keyword));
      
      if (shouldRetrieveImages) {
        try {
          const imageRetrievalResult = await this.pythonServiceClient.retrieveImagesByText(
            question,
            fileId,
            this.config.retrieval.nResults || 3
          );
          imageResults = imageRetrievalResult.images || [];
        } catch (error) {
          // Non-fatal: log but continue
          console.warn("Image retrieval failed (non-fatal):", error);
        }
      }
    } catch (error) {
      console.error("Retrieval failed:", error);
      throw error;
    }
    
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
    if (imageResults.length > 0) {
      systemPrompt += `\n\nRelevant images found: ${imageResults.length} image(s) related to the query.`;
    }
    
    // Create new messages array with system prompt at the start
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history
    ];

    yield* this.ollamaService.chatStream(messages);
  }
}

