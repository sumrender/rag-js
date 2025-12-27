import { ChromaClient, Collection } from "chromadb";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";

export class ChromaService {
  private client: ChromaClient;

  constructor(private config: Config) {
    this.client = new ChromaClient({ path: config.chroma.url });
  }

  async getCollection(): Promise<Collection> {
    try {
      return await this.client.getOrCreateCollection({
        name: this.config.chroma.collectionName
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`${ResultCodes.CHROMADB_CONNECTION_FAILED}: ${errorMessage}`);
    }
  }

  async collectionExists(): Promise<boolean> {
    try {
      const collections = await this.client.listCollections();
      return collections.includes(this.config.chroma.collectionName);
    } catch {
      return false;
    }
  }

  async getCollectionCount(collection: Collection): Promise<number> {
    try {
      const count = await collection.count();
      return count;
    } catch {
      return 0;
    }
  }
}

