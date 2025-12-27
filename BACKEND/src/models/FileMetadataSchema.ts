import mongoose, { Schema, Document } from 'mongoose';
import { FileMetadata } from './FileMetadata.js';

export interface FileMetadataDocument extends FileMetadata, Document {}

const FileMetadataSchema: Schema = new Schema({
  id: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  type: { type: String, required: true },
  createdOn: { type: String, required: true },
  path: { type: String, required: true }
}, {
    timestamps: true
});

export const FileMetadataModel = mongoose.model<FileMetadataDocument>('FileMetadata', FileMetadataSchema);
