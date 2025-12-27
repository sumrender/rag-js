import { FileMetadata } from '../models/FileMetadata.js';
import { FileMetadataModel } from '../models/FileMetadataSchema.js';

export class FileRepository {
  async save(metadata: FileMetadata): Promise<void> {
    await FileMetadataModel.create(metadata);
  }

  async getAll(): Promise<FileMetadata[]> {
    const docs = await FileMetadataModel.find().lean();
    return docs.map(doc => ({
      id: doc.id,
      name: doc.name,
      type: doc.type,
      createdOn: doc.createdOn,
      path: doc.path
    }));
  }

  async getById(id: string): Promise<FileMetadata | undefined> {
    const doc = await FileMetadataModel.findOne({ id }).lean();
    if (!doc) return undefined;
    return {
      id: doc.id,
      name: doc.name,
      type: doc.type,
      createdOn: doc.createdOn,
      path: doc.path
    };
  }
}
