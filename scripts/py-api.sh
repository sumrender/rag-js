#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON_SERVICE_DIR="$PROJECT_ROOT/BACKEND/python-service"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "[PYTHON-API] 🛑 Stopping Python service..."
    pkill -f "uvicorn main:app" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Navigate to Python service directory
cd "$PYTHON_SERVICE_DIR" || {
    echo "[PYTHON-API] Error: Could not navigate to BACKEND/python-service directory"
    exit 1
}

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[PYTHON-API] Error: Virtual environment not found. Please create it first:"
    echo "[PYTHON-API]   python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "[PYTHON-API] Activating virtual environment..."
source venv/bin/activate

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "[PYTHON-API] Error: uvicorn not found in virtual environment. Please install dependencies:"
    echo "[PYTHON-API]   pip install -r requirements.txt"
    exit 1
fi

# Set Redis environment variables (with defaults)
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}
export CACHE_TTL_QUERY_RESULTS=${CACHE_TTL_QUERY_RESULTS:-3600}
export CACHE_TTL_FAISS_RESULTS=${CACHE_TTL_FAISS_RESULTS:-7200}
export ENABLE_QUERY_CACHE=${ENABLE_QUERY_CACHE:-true}
export ENABLE_FAISS_CACHE=${ENABLE_FAISS_CACHE:-true}

# Start uvicorn server
echo "[PYTHON-API] Starting uvicorn server on http://0.0.0.0:8001..."
echo "[PYTHON-API] Redis cache: ${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
uvicorn main:app --reload --host 0.0.0.0 --port 8001 2>&1 | awk '{print "[PYTHON-API] " $0; fflush()}'

