import asyncio
import ollama
from typing import AsyncGenerator, Dict, Any, List
from ..services.retrieval_service import RetrievalService

class QueryService:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        self.ollama_service = ollama.AsyncClient()

    def _format_context_with_citations(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        if not documents:
            return ""

        formatted_parts = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i]
            citation = ""
            if metadata:
                if "pageNumber" in metadata:
                    citation = f"[Page {metadata['pageNumber']}]"
                elif "pageRange" in metadata:
                    citation = f"[Pages {metadata['pageRange']}]"

            if citation:
                formatted_parts.append(f"{citation}\n{doc}")
            else:
                formatted_parts.append(doc)

        return "\n\n---\n\n".join(formatted_parts)

    async def ask_stream(self, history: List[Dict[str, str]], file_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        last_user_message = next((msg for msg in reversed(history) if msg["role"] == "user"), None)
        if not last_user_message:
            raise ValueError("No user message found in history")

        question = last_user_message["content"]

        retrieval_result = self.retrieval_service.retrieve_text(question, file_id)

        if not retrieval_result.text_chunks:
            yield {"type": "text", "data": "I couldn't find relevant information in the documents."}
            return

        context = self._format_context_with_citations(
            [chunk.text for chunk in retrieval_result.text_chunks],
            [chunk.metadata for chunk in retrieval_result.text_chunks],
        )

        system_prompt = f"""Use the following context to answer the user's question. If the context doesn't contain enough information, you can say so or use your general knowledge, but prioritize the context.
IMPORTANT => Don't use markdown or any other formatting. Always return answer in plain text.

Context:
{context}
"""
        messages = [{"role": "system", "content": system_prompt}] + history

        stream = await self.ollama_service.chat(model='llama2', messages=messages, stream=True)

        for chunk in stream:
            if "content" in chunk["message"]:
                yield {"type": "text", "data": chunk["message"]["content"]}
