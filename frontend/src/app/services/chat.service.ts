import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { ChatMessage } from '../models/chat.model';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private messagesSubject = new BehaviorSubject<ChatMessage[]>([]);
  public messages$: Observable<ChatMessage[]> = this.messagesSubject.asObservable();

  getMessages(): ChatMessage[] {
    return this.messagesSubject.value;
  }

  addMessage(role: 'user' | 'assistant', content: string, images?: Array<{imageId: string; imageUrl: string; pageNumber?: number; score: number}>): void {
    const messages = this.messagesSubject.value;
    const newMessage: ChatMessage = {
      role,
      content,
      timestamp: new Date(),
      images
    };
    this.messagesSubject.next([...messages, newMessage]);
  }

  clearMessages(): void {
    this.messagesSubject.next([]);
  }

  updateLastMessage(content: string): void {
    const messages = this.messagesSubject.value;
    if (messages.length === 0) return;

    const lastMessage = { ...messages[messages.length - 1], content };
    const newMessages = [...messages.slice(0, -1), lastMessage];
    this.messagesSubject.next(newMessages);
  }

  appendToLastMessage(chunk: string): void {
    const messages = this.messagesSubject.value;
    if (messages.length === 0) return;

    const lastMessage = messages[messages.length - 1];
    const updatedMessage = { ...lastMessage, content: lastMessage.content + chunk };
    const newMessages = [...messages.slice(0, -1), updatedMessage];
    this.messagesSubject.next(newMessages);
  }

  addImagesToLastMessage(images: Array<{imageId: string; imageUrl: string; pageNumber?: number; score: number}>): void {
    const messages = this.messagesSubject.value;
    if (messages.length === 0) return;

    const lastMessage = messages[messages.length - 1];
    const updatedMessage = { ...lastMessage, images };
    const newMessages = [...messages.slice(0, -1), updatedMessage];
    this.messagesSubject.next(newMessages);
  }
}
