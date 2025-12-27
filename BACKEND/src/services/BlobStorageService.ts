import { BlobServiceClient, ContainerClient } from '@azure/storage-blob';

export class BlobStorageService {
  private blobServiceClient: BlobServiceClient;
  private containerName: string;
  private containerClient: ContainerClient;

  constructor(connectionString: string, containerName: string = 'documents') {
    const blobServiceClient: BlobServiceClient =
      BlobServiceClient.fromConnectionString(connectionString);

    this.blobServiceClient = blobServiceClient;
    this.containerName = containerName;
    this.containerClient = this.blobServiceClient.getContainerClient(this.containerName);
  }

  async ensureContainerExists(): Promise<void> {
    const exists: boolean = await this.containerClient.exists();

    if (!exists) {
      await this.containerClient.create({ access: 'blob' });
      console.log(`Created blob container: ${this.containerName}`);
    }
  }

  async uploadRaw(content: Buffer, fileName: string): Promise<string> {
    await this.ensureContainerExists();

    const blockBlobClient = this.containerClient.getBlockBlobClient(fileName);
    await blockBlobClient.uploadData(content);

    return blockBlobClient.name;
  }

  async download(fileName: string): Promise<string> {
    await this.ensureContainerExists();
    const blockBlobClient = this.containerClient.getBlockBlobClient(fileName);
    const downloadBlockBlobResponse = await blockBlobClient.download(0);

    return this.streamToString(
      downloadBlockBlobResponse.readableStreamBody as NodeJS.ReadableStream
    );
  }

  async downloadBuffer(fileName: string): Promise<Buffer> {
    await this.ensureContainerExists();
    const blockBlobClient = this.containerClient.getBlockBlobClient(fileName);
    const downloadBlockBlobResponse = await blockBlobClient.download(0);

    return this.streamToBuffer(
      downloadBlockBlobResponse.readableStreamBody as NodeJS.ReadableStream
    );
  }

  private async streamToString(
    readableStream: NodeJS.ReadableStream
  ): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const chunks: string[] = [];

      readableStream.on('data', (data: Buffer) => {
        chunks.push(data.toString());
      });

      readableStream.on('end', () => {
        resolve(chunks.join(''));
      });

      readableStream.on('error', reject);
    });
  }

  private async streamToBuffer(
    readableStream: NodeJS.ReadableStream
  ): Promise<Buffer> {
    return new Promise<Buffer>((resolve, reject) => {
      const chunks: Buffer[] = [];

      readableStream.on('data', (data: Buffer) => {
        chunks.push(data);
      });

      readableStream.on('end', () => {
        resolve(Buffer.concat(chunks));
      });

      readableStream.on('error', reject);
    });
  }
}
