import { Ollama } from "ollama";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";

export class OllamaService {
  private client: Ollama;

  constructor(private config: Config) {
    this.client = new Ollama({ host: config.ollama.baseUrl });
  }

  async embed(text: string): Promise<number[]> {
    try {
      const response = await this.client.embeddings({
        model: this.config.ollama.embeddingModel,
        prompt: text
      });
      return response.embedding;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`${ResultCodes.EMBEDDING_FAILED}: ${errorMessage}`);
    }
  }

  async generate(prompt: string, stream: boolean = false): Promise<string> {
    try {
      if (stream) {
        // For streaming, we'd need to handle the async iterator
        // For now, return empty string as streaming is not fully implemented
        await this.client.generate({
          model: this.config.ollama.llmModel,
          prompt,
          stream: true
        });
        return "";
      } else {
        const response = await this.client.generate({
          model: this.config.ollama.llmModel,
          prompt,
          stream: false
        });
        return response.response;
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`${ResultCodes.GENERATION_FAILED}: ${errorMessage}`);
    }
  }

  async *generateStream(prompt: string): AsyncGenerator<string> {
    try {
      const stream = await this.client.generate({
        model: this.config.ollama.llmModel,
        prompt,
        stream: true
      });

      for await (const chunk of stream) {
        yield chunk.response;
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`${ResultCodes.GENERATION_FAILED}: ${errorMessage}`);
    }
  }

  async *chatStream(messages: { role: string; content: string }[]): AsyncGenerator<string> {
    try {
      const stream = await this.client.chat({
        model: this.config.ollama.llmModel,
        messages,
        stream: true // Enable streaming
      });

      for await (const chunk of stream) {
        yield chunk.message.content;
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`${ResultCodes.GENERATION_FAILED}: ${errorMessage}`);
    }
  }
}

