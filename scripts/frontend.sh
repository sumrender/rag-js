#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
FRONTEND_DIR="$PROJECT_ROOT/FRONTEND"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "[FRONTEND] 🛑 Stopping frontend server..."
    pkill -f "npm start" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Navigate to FRONTEND directory
cd "$FRONTEND_DIR" || {
    echo "[FRONTEND] Error: Could not navigate to FRONTEND directory"
    exit 1
}

# Install dependencies
echo "[FRONTEND] 📦 Installing frontend dependencies..."
npm i

# Start frontend server
echo "[FRONTEND] 🚀 Starting Frontend Server..."
npm start 2>&1 | awk '{print "[FRONTEND] " $0; fflush()}'

