import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { FileUploadResponse, UploadProgressEvent, FileMetadata } from '../models/chat.model';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly baseUrl = 'http://localhost:3000';
  private readonly http = inject(HttpClient);

  uploadFile(file: File): Observable<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post<FileUploadResponse>(`${this.baseUrl}/file-upload`, formData)
      .pipe(catchError(this.handleError));
  }

  uploadFileWithProgress(file: File): Observable<UploadProgressEvent> {
    return new Observable<UploadProgressEvent>((observer) => {
      const formData = new FormData();
      formData.append('file', file);

      // Use fetch to upload file and get SSE stream
      fetch(`${this.baseUrl}/upload-progress`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      }).then(async (response) => {
        if (!response.ok) {
          const error = await response.json();
          observer.error(new Error(error.error || 'Upload failed'));
          return;
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          observer.error(new Error('No response body'));
          return;
        }

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                try {
                  const progressEvent = JSON.parse(data) as UploadProgressEvent;
                  observer.next(progressEvent);

                  if (progressEvent.stage === 'complete' || progressEvent.stage === 'error') {
                    observer.complete();
                    return;
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data:', e);
                }
              }
            }
          }
          observer.complete();
        } catch (error) {
          observer.error(error);
        }
      }).catch((error) => {
        observer.error(error);
      });
    });
  }

  getFiles(): Observable<FileMetadata[]> {
    return this.http.get<FileMetadata[]>(`${this.baseUrl}/files`)
      .pipe(catchError(this.handleError));
  }

  sendMessage(history: { role: string; content: string }[], fileId?: string): Observable<{type: 'text' | 'images', data: any}> {
    return new Observable<{type: 'text' | 'images', data: any}>((observer) => {
      fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ history, fileId }),
        credentials: 'include'
      }).then(async (response) => {
        if (!response.ok) {
          const error = await response.json();
          observer.error(new Error(error.error || 'Request failed'));
          return;
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          observer.error(new Error('No response body'));
          return;
        }

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                if (!data.trim()) continue;
                
                try {
                  const parsed = JSON.parse(data);
                  if (parsed.type === 'images') {
                    observer.next({ type: 'images', data: parsed.images });
                  } else if (parsed.type === 'text' || parsed.chunk) {
                    observer.next({ type: 'text', data: parsed.chunk || parsed.data });
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data:', e);
                }
              }
            }
          }
          observer.complete();
        } catch (error) {
          observer.error(error);
        }
      }).catch((error) => {
        observer.error(error);
      });
    });
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = error.error?.error || error.message || 'Server error';
    }
    
    return throwError(() => new Error(errorMessage));
  }
}
