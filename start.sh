#!/bin/bash

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Also kill any remaining npm processes to ensure cleanup
    pkill -f "npm run dev" 2>/dev/null
    pkill -f "npm start" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# In root directory
echo "🚀 Starting Docker Compose..."
docker compose up -d

# In BACKEND directory
echo "📦 Installing backend dependencies..."
cd BACKEND
npm i

echo "🚀 Starting Backend Server..."
npm run dev 2>&1 | awk '{print "[BACKEND] " $0; fflush()}' &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# In FRONTEND directory
echo "📦 Installing frontend dependencies..."
cd FRONTEND
npm i

echo "🚀 Starting Frontend Server..."
npm start 2>&1 | awk '{print "[FRONTEND] " $0; fflush()}' &
FRONTEND_PID=$!
cd ..

# Keep the script running
wait
