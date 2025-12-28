import { OllamaService } from "./OllamaService.js";
import { ChromaService } from "./ChromaService.js";
import { Config } from "../config/Config.js";
import { RedisCacheService } from "../Cache/RedisCacheService.js";

export class QueryService {
  private cache: RedisCacheService;

  constructor(
    private ollamaService: OllamaService,
    private chromaService: ChromaService,
    private config: Config
  ) {
    this.cache = new RedisCacheService();
  }

  async askSimple(question: string): Promise<string> {
    return await this.ollamaService.generate(question);
  }

  async *askStream(history: { role: string; content: string }[], fileId?: string): AsyncGenerator<string> {

    const lastUserMessage = history.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) throw new Error("No user message found");

    const question = lastUserMessage.content.trim();

    const exactCache = await this.cache.getExact(question);
    if (exactCache) {
      console.log("⚡ Exact Cache Hit");
      yield exactCache;
      return;
    }

    const queryEmbedding = await this.ollamaService.embed(question);

    const fuzzyCache = await this.cache.getFuzzy(queryEmbedding);
    if (fuzzyCache) {
      console.log("⚡ Fuzzy Cache Hit");
      yield fuzzyCache;
      return;
    }

    const collection = await this.chromaService.getCollection();

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

    const messages = [
      { role: 'system', content: systemPrompt },
      ...history
    ];

    let finalAnswer = "";

    console.log("🧠 LLM response streaming...");

    for await (const chunk of this.ollamaService.chatStream(messages)) {
      finalAnswer += chunk;
      yield chunk;
    }

    console.log("📌 Saving to cache (Exact & Fuzzy)...");
    this.cache.setExact(question, finalAnswer);
    this.cache.setFuzzy(question, finalAnswer, queryEmbedding);
  }
}

