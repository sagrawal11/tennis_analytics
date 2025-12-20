#!/bin/bash
# Start Frontend Server

cd "$(dirname "$0")/frontend"
echo "🚀 Starting Frontend Server..."
echo "📍 Frontend will be available at http://localhost:3000"
echo ""
npm run dev

