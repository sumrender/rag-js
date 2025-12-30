import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked, inject, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { ChatService } from '../../services/chat.service';
import { ChatMessage } from '../../models/chat.model';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  
  private readonly apiService = inject(ApiService);
  private readonly chatService = inject(ChatService);
  private readonly router = inject(Router);
  private readonly cdr = inject(ChangeDetectorRef);
  
  messages: ChatMessage[] = [];
  userInput = '';
  isLoading = false;
  errorMessage = '';
  private shouldScroll = false;

  files: { id: string; name: string }[] = [];
  selectedFileId = '';

  ngOnInit(): void {
    this.chatService.messages$.subscribe(messages => {
      this.messages = messages;
      this.shouldScroll = true;
    });

    this.loadFiles();
  }

  loadFiles(): void {
    this.apiService.getFiles().subscribe({
      next: (files) => {
        this.files = files;
        this.cdr.detectChanges();
      },
      error: (error) => {
        console.error('Error loading files:', error);
      }
    });
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  sendMessage(): void {
    if (!this.userInput.trim() || this.isLoading) {
      return;
    }

    const question = this.userInput.trim();
    this.userInput = '';
    this.errorMessage = '';

    // Add user message to chat
    this.chatService.addMessage('user', question);
    
    // Send to API
    this.isLoading = true;
    console.log('Sending message to API...');

    // Prepare history from current messages
    const history = this.chatService.getMessages().map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    this.apiService.sendMessage(history, this.selectedFileId).subscribe({
      next: (event) => {
        if (event.type === 'images') {
          // Store images for the current assistant message
          const messages = this.chatService.getMessages();
          if (messages.length === 0 || messages[messages.length - 1].role !== 'assistant') {
            // Create assistant message with images but no text yet
            this.chatService.addMessage('assistant', '', event.data);
            this.isLoading = false;
          } else {
            // Add images to existing assistant message
            this.chatService.addImagesToLastMessage(event.data);
          }
        } else if (event.type === 'text') {
          const messages = this.chatService.getMessages();
          const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
          
          // If this is the first chunk (isLoading is still true)
          if (this.isLoading) {
            this.isLoading = false;
            // If assistant message already exists (from images), update it with text
            if (lastMessage && lastMessage.role === 'assistant') {
              this.chatService.updateLastMessage(event.data);
            } else {
              // Create new assistant message with the first chunk
              this.chatService.addMessage('assistant', event.data);
            }
          } else {
            // Append subsequent chunks to existing assistant message
            if (lastMessage && lastMessage.role === 'assistant') {
              this.chatService.appendToLastMessage(event.data);
            } else {
              // Shouldn't happen, but create message if needed
              this.chatService.addMessage('assistant', event.data);
            }
          }
        }
        this.cdr.detectChanges(); // Force view update
      },
      error: (error) => {
        console.error('API error:', error);
        this.isLoading = false;
        this.errorMessage = error.message || 'Failed to get response';
        
        // Remove the user message if there was an error
        const messages = this.chatService.getMessages();
        if (messages.length > 0 && messages[messages.length - 1].role === 'user') {
          this.chatService.clearMessages();
          messages.slice(0, -1).forEach(msg => {
            this.chatService.addMessage(msg.role, msg.content);
          });
          this.userInput = question;
        }
        this.cdr.detectChanges();
      },
      complete: () => {
        this.isLoading = false; 
        this.cdr.detectChanges();
      }
    });
  }

  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  clearChat(): void {
    this.chatService.clearMessages();
    this.errorMessage = '';
  }

  onFileSelected(fileId: string): void {
    this.selectedFileId = fileId;
    this.clearChat();
  }

  goToUpload(): void {
    this.router.navigate(['/']);
  }

  private scrollToBottom(): void {
    try {
      if (this.messagesContainer) {
        this.messagesContainer.nativeElement.scrollTop = 
          this.messagesContainer.nativeElement.scrollHeight;
      }
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }
}
