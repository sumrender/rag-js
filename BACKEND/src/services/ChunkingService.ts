import { ResultCodes } from "../utils/ResultCodes.js";

export interface ChunkWithMetadata {
  text: string;
  startChar: number;
  endChar: number;
  pageNumber?: string;
  pageRange?: string;
}

export class ChunkingService {
  /**
   * Chunk text into smaller pieces with overlap.
   * Returns array of chunk strings.
   */
  chunkText(text: string, chunkSize: number, overlap: number): string[] {
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

  /**
   * Chunk text with page tracking for PDFs.
   * Returns array of chunks with metadata including page numbers.
   */
  chunkWithPages(
    pages: Array<{ text: string; pageNum: number }>,
    chunkSize: number,
    overlap: number
  ): ChunkWithMetadata[] {
    const chunks: ChunkWithMetadata[] = [];
    
    // Build character-to-page mapping
    let currentChar = 0;
    const charToPage: Array<{ char: number; pageNum: number }> = [];
    
    for (const page of pages) {
      const pageLength = page.text.length;
      for (let i = 0; i < pageLength; i++) {
        charToPage.push({ char: currentChar + i, pageNum: page.pageNum });
      }
      currentChar += pageLength;
    }
    
    // Combine all text
    const fullText = pages.map(p => p.text).join('');
    
    // Use the same chunking algorithm as chunkText, but track pages
    let start = 0;
    const safeOverlap = Math.min(overlap, chunkSize - 1);
    
    while (start < fullText.length) {
      const end = Math.min(start + chunkSize, fullText.length);
      const chunk = fullText.slice(start, end).trim();
      
      if (chunk.length > 0) {
        // Find pages for this chunk
        const startPage = charToPage[Math.min(start, charToPage.length - 1)]?.pageNum ?? 1;
        const endPage = charToPage[Math.min(end - 1, charToPage.length - 1)]?.pageNum ?? 1;
        
        let pageNumber: string | undefined;
        let pageRange: string | undefined;
        
        if (startPage === endPage) {
          pageNumber = startPage.toString();
        } else {
          pageRange = `${startPage}-${endPage}`;
        }
        
        chunks.push({
          text: chunk,
          startChar: start,
          endChar: end,
          pageNumber,
          pageRange
        });
      }
      
      const nextStart = end - safeOverlap;
      if (nextStart <= start) {
        start = end;
      } else {
        start = nextStart;
      }
      
      if (chunks.length > 1000000) {
        throw new Error(`${ResultCodes.TOO_MANY_CHUNKS}: Too many chunks generated. Check chunking parameters.`);
      }
    }
    
    return chunks;
  }

  /**
   * Chunk text and return chunks with character range metadata (for TXT files).
   * Uses the same chunking algorithm as chunkText but includes metadata.
   */
  chunkTextWithMetadata(
    text: string,
    chunkSize: number,
    overlap: number
  ): ChunkWithMetadata[] {
    const chunks: ChunkWithMetadata[] = [];
    let start = 0;
    
    // Ensure overlap is less than chunkSize to prevent infinite loops
    const safeOverlap = Math.min(overlap, chunkSize - 1);
    
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      const chunk = text.slice(start, end).trim();
      
      if (chunk.length > 0) {
        chunks.push({
          text: chunk,
          startChar: start,
          endChar: end
        });
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
}

