import { Router, Request, Response } from "express";
import { ServiceContainer } from "../container/ServiceContainer.js";

export function createFilesRoutes(container: ServiceContainer): Router {
  const router = Router();

  router.get("/files", async (_: Request, res: Response) => {
    res.setHeader('Cache-Control', 'no-store');
    const files = await container.fileRepository.getAll();
    res.json(files);
  });

  return router;
}

