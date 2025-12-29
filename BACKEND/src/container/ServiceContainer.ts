import { Config } from "../config/Config.js";
import { OllamaService } from "../services/OllamaService.js";
import { ChromaService } from "../services/ChromaService.js";
import { QueryService } from "../services/QueryService.js";
import { IngestionService } from "../services/IngestionService.js";
import { BlobStorageService } from "../services/BlobStorageService.js";
import { FileRepository } from "../repositories/FileRepository.js";

export class ServiceContainer {
  public readonly queryService: QueryService;
  public readonly ingestService: IngestionService;
  public readonly blobStorageService: BlobStorageService;
  public readonly fileRepository: FileRepository;

  constructor(config: Config) {
    this.blobStorageService = new BlobStorageService(
      config.azure.connectionString,
      config.azure.containerName
    );
    const ollamaService = new OllamaService(config);
    const chromaService = new ChromaService(config);
    this.queryService = new QueryService(ollamaService, chromaService, config);
    this.ingestService = new IngestionService(
      ollamaService,
      chromaService,
      config,
      this.blobStorageService
    );
    this.fileRepository = new FileRepository();

    console.log("✅ Services initialized");
  }
}

