import { OllamaService } from "./OllamaService.js";
import { ChromaService } from "./ChromaService.js";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";

export class QueryService {
  constructor(
    private ollamaService: OllamaService,
    private chromaService: ChromaService,
    private config: Config
  ) {}

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

  async *askStream(history: { role: string; content: string }[], fileId?: string): AsyncGenerator<string> {
    const lastUserMessage = history.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) {
       throw new Error("No user message found in history");
    }
    const question = lastUserMessage.content;

    const collection = await this.chromaService.getCollection();
    const queryEmbedding = await this.ollamaService.embed(question);
    
    const queryOptions: {
      queryEmbeddings: number[][];
      nResults: number;
      where?: Record<string, string>;
    } = {
      queryEmbeddings: [queryEmbedding],
      nResults: this.config.retrieval.nResults
    };

    if (fileId !== undefined && fileId !== null && fileId !== '') {
      queryOptions.where = { fileId: fileId };
    }

    const results = await collection.query(queryOptions);
    
    let context = "";
    if (results.documents?.length > 0 && results.documents[0]?.length > 0) {
      context = results.documents[0].join("\n\n---\n\n");
    }
    
    const systemPrompt = `
Use the following context to answer the user's question. If the context doesn't contain enough information, you can say so or use your general knowledge, but prioritize the context.
IMPORTANT => Don't use markdown or any other formatting. Always return answer in plain text. 

Context:
${context}
`;
    
    // Create new messages array with system prompt at the start
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history
    ];

    yield* this.ollamaService.chatStream(messages);
  }
}

