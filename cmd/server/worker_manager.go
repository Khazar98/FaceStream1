package main

import (
	"context"
	"errors"
	"log"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
	"video-analytics-system/internal/video"
)

// Xəta mesajları
var (
	ErrCameraAlreadyRunning = errors.New("camera with this ID is already running")
	ErrWorkerNotFound       = errors.New("worker not found")
)

// Hər bir RTSP axınını təmsil edən struct.
type Worker struct {
	ID       int64
	CameraID string
	RTSPLink string
	Status   string
	FPS      float32
	ctx      context.Context
	cancel   context.CancelFunc
}

// Bütün worker-ləri idarə edən mərkəzi struct.
type WorkerManager struct {
	workers      sync.Map // Eyni anda bir neçə thread-dən müraciət üçün təhlükəsiz map. Key: cameraID, Value: *Worker
	workerIDMap  sync.Map // Worker ID-yə görə axtarış üçün əlavə map. Key: workerID, Value: cameraID
	nextWorkerID int64
	batchQueue   chan FrameRequest // [BATCH] Global frame queue
}

type FrameRequest struct {
	CameraID string
	Frame    unsafe.Pointer
	Width    int
	Height   int
}

// Yeni WorkerManager instansı yaradır.
func NewWorkerManager() *WorkerManager {
	wm := &WorkerManager{
		batchQueue: make(chan FrameRequest, 500), // 4 streams × 32 batch = 128 concurrent
	}
	go wm.startBatchProcessor()
	return wm
}

// Yeni bir worker (goroutine) başladır.
func (m *WorkerManager) StartWorker(cameraID, rtspLink string) (int64, error) {
	// Eyni cameraID ilə worker-in olub-olmadığını yoxlayırıq.
	if _, ok := m.workers.Load(cameraID); ok {
		return 0, ErrCameraAlreadyRunning
	}

	// Atomik olaraq yeni unikal worker ID yaradırıq.
	workerID := atomic.AddInt64(&m.nextWorkerID, 1)

	ctx, cancel := context.WithCancel(context.Background())

	worker := &Worker{
		ID:       workerID,
		CameraID: cameraID,
		RTSPLink: rtspLink,
		Status:   "starting",
		ctx:      ctx,
		cancel:   cancel,
	}

	// Worker-i hər iki map-a əlavə edirik.
	m.workers.Store(cameraID, worker)
	m.workerIDMap.Store(workerID, cameraID)

	// Əsas video emalı məntiqini ayrı bir goroutine-də işə salırıq.
	go m.runWorkerLoop(worker)

	return worker.ID, nil
}

// Worker-i dayandırmaq üçün əsas funksiya.
func (m *WorkerManager) stopWorker(cameraID string, workerID int64) error {
	val, ok := m.workers.Load(cameraID)
	if !ok {
		return ErrWorkerNotFound
	}

	worker := val.(*Worker)

	// `context.Cancel()` çağıraraq goroutine-ə dayanma siqnalı göndəririk.
	worker.cancel()
	worker.Status = "stopped"

	// Worker-ləri map-lardan silirik.
	m.workers.Delete(cameraID)
	m.workerIDMap.Delete(worker.ID)

	return nil
}

// Camera ID-yə görə worker-i dayandırır.
func (m *WorkerManager) StopWorkerByCameraID(cameraID string) error {
	val, ok := m.workers.Load(cameraID)
	if !ok {
		return ErrWorkerNotFound
	}
	worker := val.(*Worker)
	return m.stopWorker(cameraID, worker.ID)
}

// Worker ID-yə görə worker-i dayandırır.
func (m *WorkerManager) StopWorkerByWorkerID(workerID int64) error {
	val, ok := m.workerIDMap.Load(workerID)
	if !ok {
		return ErrWorkerNotFound
	}
	cameraID := val.(string)
	return m.stopWorker(cameraID, workerID)
}

// Bütün aktiv worker-ləri dayandırır (graceful shutdown üçün).
func (m *WorkerManager) StopAllWorkers() {
	m.workers.Range(func(key, value interface{}) bool {
		cameraID := key.(string)
		log.Printf("Stopping worker for camera %s", cameraID)
		m.StopWorkerByCameraID(cameraID)
		return true
	})
}

// Aktiv worker-lərin statusunu qaytarır.
func (m *WorkerManager) GetStatus() map[string]interface{} {
	activeCameras := []map[string]interface{}{}
	m.workers.Range(func(key, value interface{}) bool {
		worker := value.(*Worker)
		activeCameras = append(activeCameras, map[string]interface{}{
			"camera_id": worker.CameraID,
			"worker_id": worker.ID,
			"status":    worker.Status,
			"fps":       worker.FPS,
		})
		return true
	})
	return map[string]interface{}{"active_cameras": activeCameras}
}

// [BATCH] Global Batch Processor
func (m *WorkerManager) startBatchProcessor() {
	const batchSize = 32              // Matches per-stream TRT buffer size
	const timeout = 8 * time.Millisecond // 8ms ≈ 120 FPS cycle

	buffer := make([]FrameRequest, 0, batchSize)
	ticker := time.NewTicker(timeout)
	defer ticker.Stop()

	for {
		select {
		case req := <-m.batchQueue:
			buffer = append(buffer, req)
			if len(buffer) >= batchSize {
				m.processBuffer(buffer)
				buffer = buffer[:0] // Reset buffer
			}
		case <-ticker.C:
			if len(buffer) > 0 {
				m.processBuffer(buffer)
				buffer = buffer[:0] // Reset buffer
			}
		}
	}
}

func (m *WorkerManager) processBuffer(requests []FrameRequest) {
	if len(requests) == 0 {
		return
	}

	count := len(requests)
	ids := make([]string, count)
	ptrs := make([]unsafe.Pointer, count)
	widths := make([]int, count)
	heights := make([]int, count)

	for i, req := range requests {
		ids[i] = req.CameraID
		ptrs[i] = req.Frame
		widths[i] = req.Width
		heights[i] = req.Height
	}

	// Call CGO Batch Function with arrays
	ProcessBatch(ids, ptrs, widths, heights)

	// Free Frames (Assuming we own them after processing)
	for _, ptr := range ptrs {
		FreeFrame(ptr)
	}
}

func (m *WorkerManager) runWorkerLoop(w *Worker) {
	log.Printf("Worker %d for camera %s started with RTSP link: %s", w.ID, w.CameraID, w.RTSPLink)
	defer log.Printf("Worker %d for camera %s has been stopped.", w.ID, w.CameraID)

	w.Status = "running"

	// RTSP Stream obyekti yaradırıq (5 retry, 5 saniye interval).
	stream := video.NewRTSPStream(w.CameraID, w.RTSPLink, 5)

	// Zero-Copy Allocator qururuq
	stream.Allocator = func(size int) (unsafe.Pointer, []byte) {
		// C++-dan yaddaş al
		ptr := AllocateFrame(size)
		if ptr == nil {
			return nil, nil
		}
		// Go slice-ə çevir (kopyalamadan)
		// unsafe.Slice requires Go 1.17+
		slice := unsafe.Slice((*byte)(ptr), size)
		return ptr, slice
	}

	if err := stream.Connect(5, 5*time.Second); err != nil {
		log.Printf("ERROR: Failed to connect to RTSP stream for camera %s: %v", w.CameraID, err)
		w.Status = "error"
		return
	}

	// Frame oxunmasını başladırıq (ayrı goroutine-də).
	stream.Start()
	defer stream.Stop()

	// FPS hesablaması
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	frameCount := 0

	frameChannel := stream.GetFrameChannel()

	for {
		select {
		case <-w.ctx.Done():
			// Stop siqnalı gəldikdə dövrdən çıxırıq.
			log.Printf("Worker %d: Stopping AI stream for camera %s", w.ID, w.CameraID)
			StopAIStream(w.CameraID)
			return

		case frame := <-frameChannel:
			if frame == nil {
				// RTSP axını kəsildi.
				log.Printf("ERROR: RTSP stream ended for camera %s", w.CameraID)
				w.Status = "error"
				m.StopWorkerByCameraID(w.CameraID)
				return
			}
			
			frameCount++

			// [SIMPLIFIED] Process frame immediately (no batch queue)
			ProcessFramePtr(w.CameraID, frame.DataPtr, frame.Width, frame.Height)
			FreeFrame(frame.DataPtr)

		case <-ticker.C:
			// FPS Update
			w.FPS = float32(frameCount)
			frameCount = 0
		}
	}
}
