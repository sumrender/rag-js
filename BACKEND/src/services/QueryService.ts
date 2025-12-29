import { OllamaService } from "./OllamaService.js";
import { ChromaService } from "./ChromaService.js";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";

type QueryType = 'summary' | 'page' | 'source' | 'detail';

interface QueryAnalysis {
  type: QueryType;
  pageNumber?: number;
}

export class QueryService {
  constructor(
    private ollamaService: OllamaService,
    private chromaService: ChromaService,
    private config: Config
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
    const collection = await this.chromaService.getCollection();
    const queryEmbedding = await this.ollamaService.embed(question);
    
    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: this.config.retrieval.nResults
    });
    
    if (results.documents === undefined || results.documents.length === 0 || results.documents[0] === undefined || results.documents[0].length === 0) {
      throw new Error(ResultCodes.NO_RELEVANT_DOCUMENTS);
    }
    
    const context = results.documents[0].join("\n\n---\n\n");
    
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

    const collection = await this.chromaService.getCollection();
    const queryAnalysis = this.detectQueryType(question);
    const queryEmbedding = await this.ollamaService.embed(question);
    
    let results: any;
    let context = "";
    let retrievedMetadata: Array<Record<string, any>> | undefined;
    
    // Build base query options - not needed anymore, handled per query type
    
    // Handle different query types
    if (queryAnalysis.type === 'summary') {
      // For summary queries, prioritize summaries
      const summaryQueryOptions: {
        queryEmbeddings: number[][];
        nResults: number;
        where?: Record<string, any>;
      } = {
        queryEmbeddings: [queryEmbedding],
        nResults: this.config.retrieval.nResultsForSummaryQueries || 1
      };
      
      // Build summary where clause with proper ChromaDB syntax
      let summaryWhere: Record<string, any>;
      if (fileId !== undefined && fileId !== null && fileId !== '') {
        summaryWhere = {
          $and: [
            { contentType: { $eq: "summary" } },
            { fileId: { $eq: fileId } }
          ]
        };
      } else {
        summaryWhere = {
          contentType: { $eq: "summary" }
        };
      }
      summaryQueryOptions.where = summaryWhere;
      
      const summaryResults = await collection.query(summaryQueryOptions);
      
      // Also get some chunks for additional context
      const chunkQueryOptions: {
        queryEmbeddings: number[][];
        nResults: number;
        where?: Record<string, any>;
      } = {
        queryEmbeddings: [queryEmbedding],
        nResults: this.config.retrieval.nResults || 3
      };
      
      // Build chunk where clause with proper ChromaDB syntax
      let chunkWhere: Record<string, any>;
      if (fileId !== undefined && fileId !== null && fileId !== '') {
        chunkWhere = {
          $and: [
            { contentType: { $eq: "chunk" } },
            { fileId: { $eq: fileId } }
          ]
        };
      } else {
        chunkWhere = {
          contentType: { $eq: "chunk" }
        };
      }
      chunkQueryOptions.where = chunkWhere;
      
      const chunkResults = await collection.query(chunkQueryOptions);
      
      // Combine results: summaries first, then chunks
      const summaryDocs = (summaryResults.documents?.[0] || []).filter((doc): doc is string => doc !== null);
      const summaryMetas = (summaryResults.metadatas?.[0] || []).filter((meta): meta is Record<string, any> => meta !== null);
      const chunkDocs = (chunkResults.documents?.[0] || []).filter((doc): doc is string => doc !== null);
      const chunkMetas = (chunkResults.metadatas?.[0] || []).filter((meta): meta is Record<string, any> => meta !== null);
      
      const allDocs: string[] = [...summaryDocs, ...chunkDocs];
      const allMetas: Array<Record<string, any>> = [...summaryMetas, ...chunkMetas];
      
      context = this.formatContextWithCitations(allDocs, allMetas);
      retrievedMetadata = allMetas;
      
    } else if (queryAnalysis.type === 'page' && queryAnalysis.pageNumber) {
      // For page queries, query more results and filter by page number in memory
      // ChromaDB's where clause doesn't support complex queries like $or or $contains
      const pageNum = queryAnalysis.pageNumber;
      const pageNumStr = pageNum.toString();
      
      // Query more results to ensure we get page-specific content
      const queryOptions: {
        queryEmbeddings: number[][];
        nResults: number;
        where?: Record<string, any>;
      } = {
        queryEmbeddings: [queryEmbedding],
        nResults: Math.max(20, (this.config.retrieval.nResultsForPageQueries || 10) * 2)
      };

      if (fileId !== undefined && fileId !== null && fileId !== '') {
        queryOptions.where = { fileId: { $eq: fileId } };
      }

      results = await collection.query(queryOptions);
      
      // Filter results by page number
      if (results.documents?.length > 0 && results.documents[0]?.length > 0) {
        const filteredDocs: string[] = [];
        const filteredMetas: Record<string, any>[] = [];
        const filteredIds: string[] = [];
        const filteredDistances: number[] = [];
        
        const docs = results.documents[0];
        const metas = results.metadatas?.[0] || [];
        const ids = results.ids?.[0] || [];
        const distances = results.distances?.[0] || [];
        
        for (let i = 0; i < docs.length; i++) {
          const meta = metas[i] || {};
          const matchesPage = 
            meta.pageNumber === pageNumStr ||
            (meta.pageRange && meta.pageRange.includes(pageNumStr)) ||
            (meta.pageRange && this.pageInRange(meta.pageRange, pageNum));
          
          if (matchesPage) {
            filteredDocs.push(docs[i]);
            filteredMetas.push(meta);
            if (ids[i]) filteredIds.push(ids[i]);
            if (distances[i] !== undefined) filteredDistances.push(distances[i]);
          }
        }
        
        // If we found page-specific results, use them; otherwise use top results
        if (filteredDocs.length > 0) {
          // Limit to requested number
          const limit = this.config.retrieval.nResultsForPageQueries || 10;
          context = this.formatContextWithCitations(
            filteredDocs.slice(0, limit),
            filteredMetas.slice(0, limit)
          );
          retrievedMetadata = filteredMetas.slice(0, limit);
        } else {
          // Fallback to top results if no page match found
          const limit = this.config.retrieval.nResultsForPageQueries || 10;
          context = this.formatContextWithCitations(
            docs.slice(0, limit),
            metas.slice(0, limit)
          );
          retrievedMetadata = metas.slice(0, limit);
        }
      }
      
    } else {
      // For detail and source queries, use standard retrieval
      const queryOptions: {
        queryEmbeddings: number[][];
        nResults: number;
        where?: Record<string, any>;
      } = {
        queryEmbeddings: [queryEmbedding],
        nResults: this.config.retrieval.nResults
      };

      if (fileId !== undefined && fileId !== null && fileId !== '') {
        queryOptions.where = { fileId: { $eq: fileId } };
      }

      results = await collection.query(queryOptions);
      
      if (results.documents?.length > 0 && results.documents[0]?.length > 0) {
        context = this.formatContextWithCitations(results.documents[0], results.metadatas?.[0]);
        retrievedMetadata = results.metadatas?.[0];
      }
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
    
    // Create new messages array with system prompt at the start
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history
    ];

    yield* this.ollamaService.chatStream(messages);
  }
}

