import express from "express";
import cors from "cors";
import { Config } from "./src/config/Config.js";
import { ServiceContainer } from "./src/container/ServiceContainer.js";
import { connectDB } from "./src/db/index.js";
import { healthCheck } from "./src/utils/healthCheck.js";
import { createChatRoutes } from "./src/routes/chat.routes.js";
import { createFilesRoutes } from "./src/routes/files.routes.js";
import { createUploadRoutes } from "./src/routes/upload.routes.js";

const app = express();

// Enable CORS for Angular frontend
app.use(cors({
  origin: 'http://localhost:4200',
  methods: ['GET', 'POST'],
  credentials: true
}));

app.use(express.json());

async function startServer(): Promise<void> {
  try {
    await connectDB();
    await healthCheck();

    const config = new Config();
    const container = new ServiceContainer(config);

    // Register routes
    app.use(createChatRoutes(container));
    app.use(createFilesRoutes(container));
    app.use(createUploadRoutes(container));

    app.listen(3000, () => {
      console.log("🚀 Server running on port 3000");
    });
  } catch (error) {
    console.error("Startup failed:", error);
    process.exit(1);
  }
}

void startServer();