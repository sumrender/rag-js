export type ProgressCallback = (info: {
  stage: 'clearing' | 'chunking' | 'summarizing' | 'embedding' | 'complete' | 'uploading' | 'uploaded' | 'start' | 'error';
  current?: number;
  total?: number;
  message: string;
  percentage?: number;
}) => void;

export interface ParsedDocument {
  text: string;
  pages?: Array<{ text: string; pageNum: number }>;
}

