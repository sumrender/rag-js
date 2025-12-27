import { Component, inject, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-file-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class FileUploadComponent {
  private readonly apiService = inject(ApiService);
  private readonly router = inject(Router);
  private readonly cdr = inject(ChangeDetectorRef);
  
  selectedFile: File | null = null;
  isDragging = false;
  isUploading = false;
  uploadSuccess = false;
  errorMessage = '';
  uploadedFileName = '';
  
  // Progress tracking
  progressStage = '';
  progressPercentage = 0;
  progressMessage = '';

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFile(input.files[0]);
    }
  }

  handleFile(file: File): void {
    this.errorMessage = '';
    this.uploadSuccess = false;
    this.progressStage = '';
    this.progressPercentage = 0;
    this.progressMessage = '';

    // Validate file type
    if (!file.name.endsWith('.txt') && !file.name.endsWith('.pdf')) {
      this.errorMessage = 'Only .txt and .pdf files are allowed';
      this.selectedFile = null;
      return;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      this.errorMessage = 'File size must be less than 10MB';
      this.selectedFile = null;
      return;
    }

    this.selectedFile = file;
  }

  uploadFile(): void {
    if (!this.selectedFile) {
      this.errorMessage = 'Please select a file first';
      return;
    }

    this.isUploading = true;
    this.errorMessage = '';
    this.progressPercentage = 0;
    this.progressMessage = 'Starting upload...';
    this.progressStage = 'start';

    this.apiService.uploadFileWithProgress(this.selectedFile).subscribe({
      next: (event) => {
        this.progressStage = event.stage;
        this.progressMessage = event.message;
        if (event.percentage !== undefined) {
          this.progressPercentage = event.percentage;
        }

        if (event.stage === 'complete') {
          this.uploadSuccess = true;
          // Delay slightly to let the user see 100%
          setTimeout(() => {
            this.router.navigate(['/chat']);
          }, 1000);
        } else if (event.stage === 'error') {
          this.isUploading = false;
          this.errorMessage = event.message || 'An error occurred during upload';
        }
        this.cdr.detectChanges();
      },
      error: (error) => {
        this.isUploading = false;
        this.errorMessage = error.message || 'Failed to upload file';
      },
      complete: () => {
        // Observer completion doesn't necessarily mean success in our flow (due to error stage),
        // but we handle success/error stages in next()
        if (this.uploadSuccess) {
          this.isUploading = false;
        }
      }
    });
  }

  goToChat(): void {
    this.router.navigate(['/chat']);
  }

  getFileSize(): string {
    if (!this.selectedFile) return '';
    const bytes = this.selectedFile.size;
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  resetUpload(): void {
    this.selectedFile = null;
    this.uploadSuccess = false;
    this.errorMessage = '';
    this.progressStage = '';
    this.progressPercentage = 0;
    this.progressMessage = '';
  }
}
