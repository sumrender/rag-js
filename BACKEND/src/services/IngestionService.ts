import crypto from "crypto";
import { PDFParse } from "pdf-parse";
import { Collection } from "chromadb";
import { OllamaService } from "./OllamaService.js";
import { ChromaService } from "./ChromaService.js";
import { Config } from "../config/Config.js";
import { FileRepository } from "../repositories/FileRepository.js";
import { BlobStorageService } from "./BlobStorageService.js";
import { ChunkingService, ChunkWithMetadata } from "./ChunkingService.js";
import { ProgressCallback, ParsedDocument } from "../models/IngestionTypes.js";

export class IngestionService {
  private chunkingService: ChunkingService;

  constructor(
    private ollamaService: OllamaService,
    private chromaService: ChromaService,
    private config: Config,
    private blobStorageService: BlobStorageService,
    private fileRepository: FileRepository = new FileRepository()
  ) {
    this.chunkingService = new ChunkingService();
  }

  /**
   * Determines the file type based on the blob name extension.
   */
  private determineFileType(blobName: string): 'pdf' | 'txt' {
    const fileExtension = blobName.split('.').pop()?.toLowerCase();
    return fileExtension === 'pdf' ? 'pdf' : 'txt';
  }

  /**
   * Estimates page boundaries by splitting text evenly across pages.
   * Used as a fallback when actual page data is not available from PDF parser.
   */
  private estimatePagesFromText(
    text: string,
    numPages: number
  ): Array<{ text: string; pageNum: number }> {
    const avgCharsPerPage = numPages > 0 ? Math.max(1, Math.floor(text.length / numPages)) : 2000;
    const pages: Array<{ text: string; pageNum: number }> = [];
    
    for (let i = 0; i < numPages; i++) {
      const start = i * avgCharsPerPage;
      const end = i === numPages - 1 ? text.length : (i + 1) * avgCharsPerPage;
      pages.push({
        text: text.slice(start, end),
        pageNum: i + 1
      });
    }
    
    return pages;
  }

  /**
   * Parses a PDF document and extracts text with page information.
   */
  private async parsePdfDocument(blobName: string): Promise<ParsedDocument> {
    const buffer = await this.blobStorageService.downloadBuffer(blobName);
    const parser = new PDFParse({ data: buffer });
    
    try {
      const result = await parser.getText();
      const text = result.text;
      
      let pages: Array<{ text: string; pageNum: number }> | undefined;
      
      // Use actual page data from pdf-parse if available
      if (result.pages && result.pages.length > 0) {
        pages = result.pages.map((page) => ({
          text: page.text,
          pageNum: page.num
        }));
      } else {
        // Fallback: estimate pages if result.pages is not available
        const numPages = result.total || 1;
        pages = this.estimatePagesFromText(text, numPages);
      }
      
      return { text, pages };
    } finally {
      await parser.destroy();
    }
  }

  /**
   * Parses a text document.
   */
  private async parseTextDocument(blobName: string): Promise<ParsedDocument> {
    const text = await this.blobStorageService.download(blobName);
    return { text };
  }

  /**
   * Creates and saves file metadata to the repository.
   */
  private async createFileMetadata(blobName: string, fileType: string): Promise<{ id: string; name: string }> {
    const fileId = crypto.randomUUID();
    const fileName = blobName;
    const fileMetadata = {
      id: fileId,
      name: fileName,
      type: fileType,
      createdOn: new Date().toISOString(),
      path: blobName
    };

    await this.fileRepository.save(fileMetadata);
    console.log(`Registered file ${fileName} with ID ${fileId}`);
    
    return { id: fileId, name: fileName };
  }

  /**
   * Chunks a document based on file type and available page information.
   */
  private chunkDocument(
    text: string,
    pages: Array<{ text: string; pageNum: number }> | undefined,
    fileType: string
  ): ChunkWithMetadata[] {
    if (fileType === 'pdf' && pages) {
      return this.chunkingService.chunkWithPages(
        pages,
        this.config.chunking.chunkSize,
        this.config.chunking.chunkOverlap
      );
    } else {
      return this.chunkingService.chunkTextWithMetadata(
        text,
        this.config.chunking.chunkSize,
        this.config.chunking.chunkOverlap
      );
    }
  }

  /**
   * Generates a document summary and stores it in the collection.
   */
  private async generateAndStoreSummary(
    text: string,
    fileId: string,
    fileName: string,
    collection: Collection
  ): Promise<void> {
    const summaryPrompt = `Generate a concise summary of the following document (keep it under 2000 characters). Include:
- Main topic and purpose
- Key points and themes
- Important details
- Overall structure
- Keep in mind that you only have the starting part of the document, so you need to summarize the document based on the starting part.
- The summary should start like: "Based on the starting part of the document, the main topic is [main topic] and the key points are [key points]."

Document:
${text.substring(0, 50000)}${text.length > 50000 ? '\n\n[... document continues ...]' : ''}`;

    const summary = await this.ollamaService.generate(summaryPrompt);
    
    // Truncate summary to safe length for embedding (max ~3000 chars to stay within token limits)
    // mxbai-embed-large typically supports ~512 tokens (~2000-4000 chars)
    const MAX_EMBEDDING_LENGTH = 2000;
    const truncatedSummary = summary.length > MAX_EMBEDDING_LENGTH 
      ? summary.substring(0, MAX_EMBEDDING_LENGTH) + '[...]'
      : summary;
    
    const summaryEmbedding = await this.ollamaService.embed(truncatedSummary);
    
    // Store the full summary in documents, but use truncated version for embedding
    await collection.add({
      ids: [`${fileId}-summary`],
      metadatas: [{ 
        fileId: fileId, 
        fileName: fileName,
        contentType: 'summary'
      }],
      documents: [summary], // Store full summary for retrieval
      embeddings: [summaryEmbedding] // Use truncated version for embedding
    });
    
    console.log(`✓ Generated and stored document summary`);
  }

  /**
   * Processes chunks by generating embeddings and storing them in the collection.
   */
  private async processAndStoreChunks(
    chunks: ChunkWithMetadata[],
    fileId: string,
    fileName: string,
    collection: Collection,
    onProgress?: ProgressCallback
  ): Promise<void> {
    for (let i = 0; i < chunks.length; i++) {
      const chunkData = chunks[i];
      const embedding = await this.ollamaService.embed(chunkData.text);
      
      // Build metadata
      const metadata: Record<string, string | number> = {
        fileId: fileId,
        fileName: fileName,
        contentType: 'chunk',
        chunkIndex: i,
        totalChunks: chunks.length,
        characterRange: `${chunkData.startChar}-${chunkData.endChar}`
      };
      
      // Add page information for PDFs
      if (chunkData.pageNumber) {
        metadata.pageNumber = chunkData.pageNumber;
      }
      if (chunkData.pageRange) {
        metadata.pageRange = chunkData.pageRange;
      }
      
      await collection.add({
        ids: [`${fileId}-chunk-${i}`],
        metadatas: [metadata],
        documents: [chunkData.text],
        embeddings: [embedding]
      });
      
      const percentage = Math.round(40 + ((i + 1) / chunks.length) * 55);
      
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
  }

  async ingest(
    blobName: string,
    onProgress?: ProgressCallback
  ): Promise<void> {
    // Get collection
    const collection = await this.chromaService.getCollection();
    
    // Determine file type and parse document
    const fileType = this.determineFileType(blobName);
    const parsedDocument = fileType === 'pdf'
      ? await this.parsePdfDocument(blobName)
      : await this.parseTextDocument(blobName);
    
    // Create and save file metadata
    const { id: fileId, name: fileName } = await this.createFileMetadata(blobName, fileType);
    
    // Chunk the document
    const chunks = this.chunkDocument(
      parsedDocument.text,
      parsedDocument.pages,
      fileType
    );
    
    console.log(`Ingesting ${chunks.length} chunks...`);
    onProgress?.({
      stage: 'chunking',
      total: chunks.length,
      message: `Split document into ${chunks.length} chunks`,
      percentage: 20
    });
    
    // Generate and store document summary
    onProgress?.({
      stage: 'summarizing',
      message: 'Generating document summary...',
      percentage: 30
    });
    await this.generateAndStoreSummary(
      parsedDocument.text,
      fileId,
      fileName,
      collection
    );
    
    // Process and store chunks
    await this.processAndStoreChunks(
      chunks,
      fileId,
      fileName,
      collection,
      onProgress
    );
    
    // Report completion
    onProgress?.({
      stage: 'complete',
      total: chunks.length,
      message: `Successfully ingested ${chunks.length} chunks`,
      percentage: 100
    });
  }
}
