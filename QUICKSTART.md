# FaceStream - Quick Start Guide

## Overview
FaceStream is an AI-powered video analytics system that detects faces in video streams using YOLOv12 and TensorRT, captures the best frames, and streams results to Kafka.

**Model Used:** YOLOv12n-face (7.5 MB TensorRT engine)
**Inference Speed:** 20-30ms per frame
**Output Logs:** server.log

---

## Pre-Build Setup

### 1. Verify Dependencies
```bash
# Check for required libraries
ls -lh models/yolov12n-face.engine
ls -lh onnxruntime-linux-x64-1.20.1/lib/
ldconfig -p | grep -E "opencv|cudart|onnxruntime"
```

### 2. Check Kafka Connectivity
```bash
# Verify Kafka brokers are accessible
for broker in 10.13.3.100:9092 10.13.3.99:9092 10.13.3.101:9092; do
    nc -zv ${broker%:*} ${broker#*:}
done
```

---

## Building the Application

### Option 1: Using the Startup Script (Recommended)
```bash
./run_app.sh
```

### Option 2: Manual Build
```bash
# Clean previous builds
make clean

# Build both C++ library and Go server
make build

# Run the server
./build/video_server
```

---

## Running Face Detection

### Option 1: Using Camera Controller Script

#### Start Detection on Edrik Hansiki Stream
```bash
./control_camera.sh start "rtsp://edrik-hansiki/stream" "edrik-hansiki-01"
```

Example with UUID:
```bash
./control_camera.sh start "rtsp://192.168.1.100:554/stream" "edrik-hansiki" "550e8400-e29b-41d4-a716-446655440000"
```

#### Check Detection Status
```bash
./control_camera.sh status
```

Output:
```json
{
  "active_cameras": [
    {
      "camera_id": "edrik-hansiki-01",
      "worker_id": 1,
      "status": "running",
      "fps": 28.5
    }
  ]
}
```

#### Stop Detection
```bash
./control_camera.sh stop "550e8400-e29b-41d4-a716-446655440000"
```

---

### Option 2: Using Direct HTTP Requests

#### Start Face Detection
```bash
# In one terminal, run the server:
./build/video_server

# In another terminal, start detection on Edrik Hansiki camera:
curl -X POST http://localhost:8050/camera/start \
  -H "Content-Type: application/json" \
  -d '{
    "rtsp_link": "rtsp://edrik-hansiki.local/stream",
    "camera_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

Response:
```json
{
  "worker_id": 1,
  "status": "running"
}
```

#### Get Status
```bash
curl -X GET http://localhost:8050/camera/status
```

#### Stop Detection
```bash
curl -X POST http://localhost:8050/camera/stop \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

---

## Monitoring Logs

### Live Log Monitoring
```bash
# Watch logs in real-time
tail -f server.log

# Search for face detections
grep "detection_time" server.log
grep "best_shot" server.log

# Count detections
grep -c "confidence" server.log
```

### Log File Location
- **File:** `server.log`
- **Location:** In the application root directory
- **Format:** Timestamp + Level + Message
- **Auto-appending:** Logs append to existing file on restart

### Sample Log Output
```
2026-02-16 19:30:42 [main.go:25] ========================================
2026-02-16 19:30:42 [main.go:26] FaceStream AI Video Analytics System
2026-02-16 19:30:42 [main.go:27] ========================================
2026-02-16 19:30:42 Using AI model: /home/admiral/Khazar/CProjects/FaceStream1/models/yolov12n-face.engine
2026-02-16 19:30:42 [main.go:59] Kafka Brokers: [10.13.3.100:9092 10.13.3.99:9092 10.13.3.101:9092]
2026-02-16 19:30:43 [main.go:80] Started camera edrik-hansiki-01 with worker ID 1
2026-02-16 19:30:44 Processing frames at 28.5 FPS - Best shot quality: 85.2
2026-02-16 19:30:45 Face detection: confidence=0.95, age_estimate=32, gender=male
```

---

## Configuration

### Environment Variables

```bash
# Set custom model path
export MODEL_PATH="./models/yolov12n-face.engine"

# Set Kafka brokers (comma-separated, no spaces)
export KAFKA_BROKERS="10.13.3.100:9092,10.13.3.99:9092,10.13.3.101:9092"

# Run with custom config
./run_app.sh "$MODEL_PATH" "$KAFKA_BROKERS"
```

### Server Settings
| Setting | Value |
|---------|-------|
| Server Port | 8050 |
| Frame Batch Size | 128 frames |
| Batch Timeout | 30ms |
| Inactivity Timeout | 2 seconds |
| Max Retries (RTSP) | 5 |
| Retry Interval | 5 seconds |

---

## Complete Workflow: Edrik Hansiki Face Detection

### Step 1: Terminal 1 - Start Server
```bash
cd /home/admiral/Khazar/CProjects/FaceStream1
./run_app.sh
```

Output:
```
========================================
FaceStream - AI Video Analytics System
========================================

Configuration:
  Workspace:      /home/admiral/Khazar/CProjects/FaceStream1
  Model Path:     ./models/yolov12n-face.engine
  Kafka Brokers:  10.13.3.100:9092,10.13.3.99:9092,10.13.3.101:9092
  Log File:       server.log

✓ Model file verified: 7.5M ./models/yolov12n-face.engine
✓ Application already built

========================================
Starting FaceStream Server
========================================

API Endpoints:
  POST   http://localhost:8050/camera/start
  POST   http://localhost:8050/camera/stop
  GET    http://localhost:8050/camera/status
Press Ctrl+C to stop the server
```

### Step 2: Terminal 2 - Start Detection on Edrik Hansiki
```bash
./control_camera.sh start "rtsp://edrik-hansiki/stream" "edrik-hansiki-01"
```

### Step 3: Monitor Detection
```bash
# Terminal 3 - Watch logs in real-time
tail -f server.log

# Terminal 4 - Check status periodically
while true; do ./control_camera.sh status; sleep 5; done
```

### Step 4: Stop Detection
```bash
./control_camera.sh stop "550e8400-e29b-41d4-a716-446655440000"
# Or use camera name
./control_camera.sh stop "edrik-hansiki-01"
```

---

## Troubleshooting

### Issue: "Failed to open log file"
```bash
# Fix: Ensure write permissions
touch server.log
chmod 644 server.log
```

### Issue: "Model not found at..." 
```bash
# Fix: Verify model exists
ls -lh models/yolov12n-face.engine
# If missing, check MODEL_PATH environment variable
echo $MODEL_PATH
```

### Issue: "Failed to connect to Kafka"
```bash
# Fix: Check Kafka broker connectivity
curl telnet://10.13.3.100:9092
# Or verify env variable
echo $KAFKA_BROKERS
```

### Issue: "Stream connection failed"
```bash
# Fix: Verify RTSP URL is accessible
ffplay "rtsp://edrik-hansiki/stream"
# Add retries in request
```

### Issue: Low FPS
```bash
# Check GPU utilization
nvidia-smi watch
# Reduce batch size if memory limited
# Check network bandwidth for RTSP
```

---

## Performance Metrics

### Expected Performance
| Metric | Value |
|--------|-------|
| Model Load Time | ~1-2 seconds |
| Frame Processing | 20-30ms |
| Best Shot FPS | 25-30 FPS |
| Memory Usage | ~800-1200 MB |
| GPU Memory | ~600-800 MB |

### Monitoring Performance
```bash
# GPU stats
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits -l 1

# Kafka message rate
# Subscribe to topics and count
```

---

## API Reference

### POST /camera/start
Start face detection on an RTSP stream.

**Request:**
```json
{
  "rtsp_link": "rtsp://edrik-hansiki/stream",
  "camera_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "worker_id": 1,
  "status": "running"
}
```

**Errors:**
- `400` - Invalid request body
- `409` - Camera already running
- `500` - Internal server error

---

### POST /camera/stop
Stop face detection.

**Request:**
```json
{
  "camera_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Or by worker ID:
```json
{
  "worker_id": "1"
}
```

**Response:**
```json
{
  "status": "stopped"
}
```

---

### GET /camera/status
Get status of all active cameras.

**Response:**
```json
{
  "active_cameras": [
    {
      "camera_id": "550e8400-e29b-41d4-a716-446655440000",
      "worker_id": 1,
      "status": "running",
      "fps": 28.5
    }
  ]
}
```

---

## Output: Kafka Messages

### Topic: gate-metric-detections
```json
{
  "camera_id": "edrik-hansiki-01",
  "detection_time": "2026-02-16T19:30:45Z",
  "image_base64": "...",
  "image_base64_padded": "...",
  "confidence": 0.95,
  "age_estimate": 32,
  "gender_estimate": "male",
  "quality_metadata": {
    "sharpness": 0.88,
    "brightness": 0.75,
    "contrast": 0.82
  }
}
```

---

## Shutdown

### Graceful Shutdown
```bash
# Press Ctrl+C in the server terminal
# Server will:
# 1. Stop all active workers
# 2. Close Kafka connection
# 3. Cleanup C++ resources
# 4. Exit cleanly
```

### View Shutdown Logs
```bash
tail -20 server.log | grep -E "Shutting|stopped"
```

---

## Support

For issues or questions:
1. Check logs: `tail -f server.log`
2. Verify model path: `ls -lh models/yolov12n-face.engine`
3. Test API: `curl http://localhost:8050/camera/status`
4. Check Kafka: Verify broker connectivity

---

## Next Steps

- [ ] Test face detection from Edrik Hansiki stream
- [ ] Monitor Kafka messages
- [ ] Analyze face detection accuracy
- [ ] Configure best shot storage
- [ ] Setup alerting for high-confidence detections
