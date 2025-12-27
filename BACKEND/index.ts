import { Config } from "./src/config/Config.js";
import { OllamaService } from "./src/services/OllamaService.js";
import { ChromaService } from "./src/services/ChromaService.js";
import { QueryService } from "./src/services/QueryService.js";
import { IngestionService } from './src/services/IngestionService.js'
import express, { Request, Response } from "express";
import { uploadTxt } from "./src/middleware/upload.js"
import cors from "cors";
import { FileRepository } from "./src/repositories/FileRepository.js";
import { connectDB, checkDbConnection } from "./src/db/index.js";
import { BlobStorageService } from "./src/services/BlobStorageService.js";

const app = express();

// Enable CORS for Angular frontend
app.use(cors({
  origin: 'http://localhost:4200',
  methods: ['GET', 'POST'],
  credentials: true
}));

app.use(express.json());

let ingestService: IngestionService | null = null;
let queryService: QueryService | null = null;
let blobStorageService: BlobStorageService | null = null;
const fileRepository = new FileRepository();

// 🔹 Initialize services FIRST
function main(): void {
  const config = new Config();

  blobStorageService = new BlobStorageService(config.azure.connectionString, config.azure.containerName);
  const ollamaService = new OllamaService(config);
  const chromaService = new ChromaService(config);
  queryService = new QueryService(
    ollamaService,
    chromaService,
    config
  );
  ingestService = new IngestionService(ollamaService, chromaService, config, blobStorageService);


  console.log("✅ Services initialized");
}

// 🔹 API endpoint
app.post("/chat", async (req: Request, res: Response): Promise<void> => {
  if (!queryService) {
    res.status(503).json({ error: "Server not ready yet" });
    return;
  }

  const body = req.body as { history?: { role: string; content: string }[]; fileId?: string };
  const history = body?.history;
  const fileId = body?.fileId;

  if (!history || !Array.isArray(history) || history.length === 0) {
    res.status(400).json({ error: "Chat history is required" });
    return;
  }

  try {
    // Set SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "http://localhost:4200");
    res.setHeader("Access-Control-Allow-Credentials", "true");

    const stream = queryService.askStream(history, fileId);

    for await (const chunk of stream) {
      res.write(`data: ${JSON.stringify({ chunk })}\n\n`);
    }
    
    res.end();
  } catch (error) {
    console.error("Chat error:", error);
    // If headers already sent, we can't send JSON error, but we can close stream
    if (!res.headersSent) {
      res.status(500).json({ error: "Error processing the question" });
    } else {
      res.end();
    }
  }
});

app.get("/files", async (_: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store');
  const files = await fileRepository.getAll();
  res.json(files);
});

app.post(
  "/file-upload",
  uploadTxt.single("file"),
  async (req: Request, res: Response): Promise<void> => {
    if (!req.file) {
      res.status(400).json({ error: "TXT or PDF file is required" });
      return;
    }

    if (!ingestService) {
      res.status(503).json({ error: "ingestService not ready yet" });
      return;
    }
    if (!blobStorageService) {
      res.status(503).json({ error: "blobStorageService not ready yet" });
      return;
    }

    // multer.memoryStorage() does not provide req.file.filename; use originalname and a unique prefix.
    const safeOriginalName = (req.file.originalname || "upload")
      .replaceAll("/", "_")
      .replaceAll("\\", "_");
    const blobName = `${Date.now()}-${safeOriginalName}`;

    try {
      const uploadedBlobName = await blobStorageService.uploadRaw(req.file.buffer, blobName);

      ingestService.ingest(uploadedBlobName).catch((error: unknown) => {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error("Ingestion error:", errorMessage);
      });

      res.json({
        message: "File uploaded successfully. Wait a while for the ingestion to complete.",
        fileName: safeOriginalName,
        blobName: uploadedBlobName,
        size: req.file.size,
      });
    } catch (error) {
      console.error("Upload failed:", error);
      res.status(500).json({ error: "Failed to upload file" });
    }
  }
);

// SSE endpoint for real-time progress updates
app.post(
  "/upload-progress",
  uploadTxt.single("file"),
  async (req: Request, res: Response): Promise<void> => {
    if (!req.file) {
      res.status(400).json({ error: "TXT or PDF file is required" });
      return;
    }

    if (!ingestService) {
      res.status(503).json({ error: "ingestService not ready yet" });
      return;
    }
    if (!blobStorageService) {
      res.status(503).json({ error: "blobStorageService not ready yet" });
      return;
    }

    const safeOriginalName = (req.file.originalname || "upload")
      .replaceAll("/", "_")
      .replaceAll("\\", "_");
    const blobName = `${Date.now()}-${safeOriginalName}`;

    // Set SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "http://localhost:4200");
    res.setHeader("Access-Control-Allow-Credentials", "true");

    // Send initial connection event
    res.write(`data: ${JSON.stringify({ stage: 'start', message: 'Starting upload...', percentage: 0 })}\n\n`);

    try {
      res.write(`data: ${JSON.stringify({ stage: 'uploading', message: 'Uploading to storage...', percentage: 0 })}\n\n`);
      const uploadedBlobName = await blobStorageService.uploadRaw(req.file.buffer, blobName);
      res.write(`data: ${JSON.stringify({ stage: 'uploaded', message: 'Upload complete. Starting ingestion...', percentage: 0, blobName: uploadedBlobName })}\n\n`);

      // Start ingestion with progress callback
      ingestService.ingest(uploadedBlobName, (progressInfo) => {
        res.write(`data: ${JSON.stringify(progressInfo)}\n\n`);

        if (progressInfo.stage === 'complete') {
          res.end();
        }
      }).catch((error: unknown) => {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error("Ingestion error:", errorMessage);

        res.write(`data: ${JSON.stringify({ 
          stage: 'error', 
          message: errorMessage,
          percentage: 0
        })}\n\n`);
        res.end();
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error("Upload failed:", errorMessage);
      res.write(`data: ${JSON.stringify({ stage: 'error', message: errorMessage, percentage: 0 })}\n\n`);
      res.end();
    }
  }
);

async function healthCheck(): Promise<void> {
  const ollamaResponse = await fetch("http://localhost:11434/api/tags");
  if (!ollamaResponse.ok) {
    throw new Error("Ollama is not running");
  }
  const chromaResponse = await fetch("http://localhost:8000/api/v2/heartbeat");
  if (!chromaResponse.ok) {
    throw new Error("Chroma is not running");
  }
  const dbConnected = await checkDbConnection();
  if (!dbConnected) {
    throw new Error("MongoDB is not connected");
  }
  console.log("Ollama, Chroma, and MongoDB are running");
}


async function startServer(): Promise<void> {
  try {
    await connectDB();
    await healthCheck();
    main();

    app.listen(3000, () => {
      console.log("🚀 Server running on port 3000");
    });
  } catch (error) {
    console.error("Startup failed:", error);
    process.exit(1);
  }
}

void startServer();