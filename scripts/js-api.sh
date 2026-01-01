#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BACKEND_DIR="$PROJECT_ROOT/BACKEND"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "[BACKEND] 🛑 Stopping backend server..."
    pkill -f "npm run dev" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Navigate to BACKEND directory
cd "$BACKEND_DIR" || {
    echo "[BACKEND] Error: Could not navigate to BACKEND directory"
    exit 1
}

# Install dependencies
echo "[BACKEND] 📦 Installing backend dependencies..."
npm i

# Start backend server
echo "[BACKEND] 🚀 Starting Backend Server..."
npm run dev 2>&1 | awk '{print "[BACKEND] " $0; fflush()}'

