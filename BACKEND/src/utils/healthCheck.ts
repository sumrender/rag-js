import { checkDbConnection } from "../db/index.js";

export async function healthCheck(): Promise<void> {
  const ollamaResponse = await fetch("http://localhost:11434/api/tags");
  if (!ollamaResponse.ok) {
    throw new Error("Ollama is not running");
  }
  const dbConnected = await checkDbConnection();
  if (!dbConnected) {
    throw new Error("MongoDB is not connected");
  }
  console.log("Ollama and MongoDB are running");
}

