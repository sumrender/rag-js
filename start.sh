#!/bin/bash

# docker compose down --volumes
# To remove the volumes and start fresh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $PYTHON_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null

    echo "🛑 Cleaning docker compose..."
    docker compose down --volumes
    # Also kill any remaining processes to ensure cleanup
    pkill -f "npm run dev" 2>/dev/null
    pkill -f "npm start" 2>/dev/null
    pkill -f "uvicorn main:app" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Ensure scripts are executable
chmod +x "$SCRIPTS_DIR/js-api.sh" 2>/dev/null
chmod +x "$SCRIPTS_DIR/py-api.sh" 2>/dev/null
chmod +x "$SCRIPTS_DIR/frontend.sh" 2>/dev/null

# In root directory
echo "🚀 Starting Docker Compose..."
docker compose up -d

# Start backend server
echo "🚀 Starting Backend Server..."
"$SCRIPTS_DIR/js-api.sh" &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "🚀 Starting Frontend Server..."
"$SCRIPTS_DIR/frontend.sh" &
FRONTEND_PID=$!

# Wait a moment for Python service to start
sleep 2

# Start Python service
echo "🚀 Starting Python Service..."
"$SCRIPTS_DIR/py-api.sh" &
PYTHON_PID=$!

# Keep the script running
wait
