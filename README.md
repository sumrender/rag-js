# Simple RAG

### System Requirements

- **Node.js** ≥ 18
- **Docker** (for ChromaDB)
- **Ollama** installed and running locally

### Install Ollama

Visit [ollama.ai](https://ollama.ai) to install Ollama for your platform.

### Required Models

Ensure you have the following models installed in Ollama:

```bash
# Check installed models
ollama list

# Install required models if missing
ollama pull gemma3
ollama pull mxbai-embed-large
```

###
Once ollama is running locally.
do `docker compose up`, then run scripts from `scripts` folder
1. Run `./start.sh` in root directory
2. Run