import { createClient } from "redis";
import crypto from "crypto";

export class RedisCacheService {
  private client;

  constructor() {
    this.client = createClient({ url: "redis://localhost:6379" });
    this.client.connect();
  }

  private hash(text: string) {
    return crypto.createHash("sha256").update(text.trim().toLowerCase()).digest("hex");
  }

  // ---------------- EXACT CACHE ----------------
  async getExact(question: string) {
    const key = `cache:exact:${this.hash(question)}`;
    return await this.client.get(key);
  }

  async setExact(question: string, answer: string) {
    const key = `cache:exact:${this.hash(question)}`;
    await this.client.set(key, answer, { EX: 60 * 60 * 24 }); // TTL 24h
  }

  // ---------------- FUZZY CACHE USING VECTOR ----------------

  private async initializeIndex() {
    try {
      await this.client.ft.create('idx:cache', {
        '$.embedding': {
          type: 'VECTOR' as any,
          AS: 'embedding',
          ALGORITHM: 'HNSW',
          TYPE: 'FLOAT32',
          DIM: 1024,
          DISTANCE_METRIC: 'COSINE'
        } as any,
        '$.question': {
          type: 'TEXT',
          AS: 'question'
        },
        '$.answer': {
          type: 'TEXT',
          AS: 'answer'
        }
      }, {
        ON: 'JSON',
        PREFIX: 'cache:fuzzy:'
      });
      console.log("✅ Redis Vector Index Created");
    } catch (e: any) {
      if (e.message === 'Index already exists') {

      } else {
        console.error("❌ Redis Index Creation Error:", e);
      }
    }
  }

  async getFuzzy(embedding: number[]): Promise<string | null> {
    try {
      // Lazy initialization
      await this.initializeIndex();

      const results = await this.client.ft.search('idx:cache', `*=>[KNN 1 @embedding $BLOB AS score]`, {
        PARAMS: {
          BLOB: Buffer.from(new Float32Array(embedding).buffer)
        },
        RETURN: ['score', '$.answer', '$.question'],
        DIALECT: 2
      }) as any;

      console.log("Redis Fuzzy Search Results:", results);

      if (results.total > 0 && results.documents.length > 0) {
        const doc = results.documents[0];

        const question = doc.value.question || doc.value['$.question'];
        const answer = doc.value.answer || doc.value['$.answer'];
        const score = doc.value.score as unknown as number;

        console.log(`🔎 Fuzzy Match found. Score: ${score}, Question: "${question}"`);

        // Cosine distance: 0 is identical, 1 is opposite.
        // Tightening threshold to 0.2 (Similarity > 0.8) to avoid false positives
        if (score < 0.3) {
          return answer as string;
        } else {
          console.log(`⚠️ Match found but score ${score} is >= 0.3 (Limit)`);
        }
      } else {
        console.log("⚠️ No fuzzy match found in Redis");
      }
      return null;
    } catch (e) {
      console.error("Fuzzy Search Error:", e);
      return null;
    }
  }

  async setFuzzy(question: string, answer: string, embedding: number[]) {
    const key = `cache:fuzzy:${crypto.randomUUID()}`;
    await this.client.json.set(key, '$', {
      question,
      answer,
      embedding: embedding
    });

    await this.client.expire(key, 60 * 60 * 24);
  }
}
