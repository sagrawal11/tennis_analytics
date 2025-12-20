#!/bin/bash
# Start Backend Server

cd "$(dirname "$0")/backend"
source ../tennis_env/bin/activate
echo "🚀 Starting Backend Server..."
echo "📍 Backend will be available at http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --port 8000

