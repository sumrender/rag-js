import crypto from "crypto";
import { PDFParse } from "pdf-parse";
import { OllamaService } from "./OllamaService.js";
import { ChromaService } from "./ChromaService.js";
import { Config } from "../config/Config.js";
import { ResultCodes } from "../utils/ResultCodes.js";
import { FileRepository } from "../repositories/FileRepository.js";
import { BlobStorageService } from "./BlobStorageService.js";

export class IngestionService {
  constructor(
    private ollamaService: OllamaService,
    private chromaService: ChromaService,
    private config: Config,
    private blobStorageService: BlobStorageService,
    private fileRepository: FileRepository = new FileRepository()
  ) {}

  private chunkText(text: string, chunkSize: number, overlap: number): string[] {
    const chunks: string[] = [];
    let start = 0;
    
    // Ensure overlap is less than chunkSize to prevent infinite loops
    const safeOverlap = Math.min(overlap, chunkSize - 1);
    
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      const chunk = text.slice(start, end).trim();
      
      if (chunk.length > 0) {
        chunks.push(chunk);
      }
      
      // Move start forward, ensuring we always make progress
      const nextStart = end - safeOverlap;
      if (nextStart <= start) {
        // Safety check: ensure we always advance
        start = end;
      } else {
        start = nextStart;
      }
      
      // Safety check: prevent infinite loops
      if (chunks.length > 1000000) {
        throw new Error(`${ResultCodes.TOO_MANY_CHUNKS}: Too many chunks generated. Check chunking parameters.`);
      }
    }
    
    return chunks;
  }

  async ingest(
    blobName: string,
    onProgress?: (info: {
      stage: 'clearing' | 'chunking' | 'embedding' | 'complete';
      current?: number;
      total?: number;
      message: string;
      percentage?: number;
    }) => void
  ): Promise<void> {
    const collection = await this.chromaService.getCollection();
    
    // Determine file type from blob name
    const fileExtension = blobName.split('.').pop()?.toLowerCase();
    const fileType = fileExtension === 'pdf' ? 'pdf' : 'txt';
    
    let text: string;
    
    if (fileType === 'pdf') {
      // Download as buffer and parse PDF
      const buffer = await this.blobStorageService.downloadBuffer(blobName);
      const parser = new PDFParse({ data: buffer });
      try {
        const result = await parser.getText();
        text = result.text;
      } finally {
        await parser.destroy();
      }
    } else {
      // Download as text
      text = await this.blobStorageService.download(blobName);
    }
    
    const chunks = this.chunkText(text, this.config.chunking.chunkSize, this.config.chunking.chunkOverlap);
    
    // Generate File Metadata
    const fileId = crypto.randomUUID();
    const fileName = blobName;
    const fileMetadata = {
      id: fileId,
      name: fileName,
      type: fileType,
      createdOn: new Date().toISOString(),
      path: blobName
    };

    // Save metadata
    await this.fileRepository.save(fileMetadata);
    console.log(`Registered file ${fileName} with ID ${fileId}`);

    console.log(`Ingesting ${chunks.length} chunks...`);
    onProgress?.({
      stage: 'chunking',
      total: chunks.length,
      message: `Split document into ${chunks.length} chunks`,
      percentage: 0
    });
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await this.ollamaService.embed(chunk);
      
      await collection.add({
        ids: [`${fileId}-chunk-${i}`],
        metadatas: [{ fileId: fileId, fileName: fileName }],
        documents: [chunk],
        embeddings: [embedding]
      });
      
      const percentage = Math.round(((i + 1) / chunks.length) * 100);
      
      if ((i + 1) % 10 === 0 || i === chunks.length - 1) {
        console.log(`Progress: ${i + 1}/${chunks.length} chunks`);
        onProgress?.({
          stage: 'embedding',
          current: i + 1,
          total: chunks.length,
          message: `Processing chunk ${i + 1}/${chunks.length}`,
          percentage
        });
      }
    }
    
    console.log(`✓ Ingested ${chunks.length} chunks successfully`);
    onProgress?.({
      stage: 'complete',
      total: chunks.length,
      message: `Successfully ingested ${chunks.length} chunks`,
      percentage: 100
    });
  }
}
