import { Router, Request, Response } from "express";
import { ServiceContainer } from "../container/ServiceContainer.js";

export function createChatRoutes(container: ServiceContainer): Router {
  const router = Router();

  router.post("/chat", async (req: Request, res: Response): Promise<void> => {
    const body = req.body as { history?: { role: string; content: string }[]; fileId?: string };
    const history = body?.history;
    const fileId = body?.fileId;

    if (!history || !Array.isArray(history) || history.length === 0) {
      res.status(400).json({ error: "Chat history is required" });
      return;
    }

    try {
      // Set SSE headers
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const stream = container.queryService.askStream(history, fileId);

      for await (const event of stream) {
        if (event.type === 'images') {
          res.write(`data: ${JSON.stringify({ type: 'images', images: event.data })}\n\n`);
        } else if (event.type === 'text') {
          res.write(`data: ${JSON.stringify({ type: 'text', chunk: event.data })}\n\n`);
        }
      }
      
      res.end();
    } catch (error) {
      console.error("Chat error:", error);
      // If headers already sent, we can't send JSON error, but we can close stream
      if (!res.headersSent) {
        res.status(500).json({ error: "Error processing the question" });
      } else {
        res.end();
      }
    }
  });

  return router;
}

