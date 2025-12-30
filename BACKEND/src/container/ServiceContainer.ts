import { Config } from "../config/Config.js";
import { OllamaService } from "../services/OllamaService.js";
import { QueryService } from "../services/QueryService.js";
import { BlobStorageService } from "../services/BlobStorageService.js";
import { FileRepository } from "../repositories/FileRepository.js";
import { PythonServiceClient } from "../services/PythonServiceClient.js";

export class ServiceContainer {
  public readonly queryService: QueryService;
  public readonly blobStorageService: BlobStorageService;
  public readonly fileRepository: FileRepository;
  public readonly pythonServiceClient: PythonServiceClient;

  constructor(config: Config) {
    this.blobStorageService = new BlobStorageService(
      config.azure.connectionString,
      config.azure.containerName
    );
    const ollamaService = new OllamaService(config);
    this.pythonServiceClient = new PythonServiceClient(config);
    this.queryService = new QueryService(ollamaService, config, this.pythonServiceClient);
    this.fileRepository = new FileRepository();

    console.log("✅ Services initialized");
  }
}

