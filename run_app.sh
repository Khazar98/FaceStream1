#!/bin/bash

# FaceStream AI Video Analytics - Application Launcher
# Usage: ./run_app.sh [MODEL_PATH] [KAFKA_BROKERS]

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Configuration
MODEL_PATH="${1:-.}/models/yolov12n-face.engine"
KAFKA_BROKERS="${2:-10.13.3.100:9092,10.13.3.99:9092,10.13.3.101:9092}"
LOG_FILE="server.log"

echo "========================================"
echo "FaceStream - AI Video Analytics System"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Workspace:      $WORKSPACE_DIR"
echo "  Model Path:     $MODEL_PATH"
echo "  Kafka Brokers:  $KAFKA_BROKERS"
echo "  Log File:       $LOG_FILE"
echo ""

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "✓ Model file verified: $(ls -lh $MODEL_PATH | awk '{print $5, $9}')"
echo ""

# Set environment variables
export MODEL_PATH="$MODEL_PATH"
export KAFKA_BROKERS="$KAFKA_BROKERS"

# [FIX] Add library paths for runtime linking
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WORKSPACE_DIR/pkg/cpp_processor:$WORKSPACE_DIR/onnxruntime-linux-x64-1.20.1/lib"

# Check if build directory exists, if not build
if [ ! -f "build/video_server" ]; then
    echo "Building application..."
    make clean build
    echo "✓ Build complete"
else
    echo "✓ Application already built"
fi

echo ""
echo "========================================"
echo "Starting FaceStream Server"
echo "========================================"
echo ""
echo "API Endpoints:"
echo "  POST   http://localhost:8050/camera/start"
echo "  POST   http://localhost:8050/camera/stop"
echo "  GET    http://localhost:8050/camera/status"
echo ""
echo "Example: Start face detection from RTSP stream"
echo '  curl -X POST http://localhost:8050/camera/start \'
echo '    -H "Content-Type: application/json" \'
echo '    -d "{\"rtsp_link\":\"rtsp://example.com/stream\",\"camera_id\":\"550e8400-e29b-41d4-a716-446655440000\"}"'
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Run the application
./build/video_server
