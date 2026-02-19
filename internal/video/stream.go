package video

/*
#cgo pkg-config: libavcodec libavformat libavutil libswscale
#include "../../pkg/rtsp_reader/rtsp_reader.c"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"log"
	"time"
	"unsafe"
)

// Frame RTSP axınından oxunan video kadrını təmsil edir.
type Frame struct {
	Data      []byte         // Slice interface for Go access
	DataPtr   unsafe.Pointer // Pointer to C++ memory
	Width     int
	Height    int
	Timestamp time.Time
}

// AllocatorFunc is a function that returns a buffer of a given size.
type AllocatorFunc func(size int) (unsafe.Pointer, []byte)

// RTSPStream RTSP axınını idarə edən struct.
type RTSPStream struct {
	URL       string
	CameraID  string
	frameChan chan *Frame
	stopChan  chan struct{}
	isRunning bool
	Allocator AllocatorFunc // Optional custom allocator (e.g. from C++ pool)
	
	// FFMpeg Context
	avCtx *C.RtspContext
}

// NewRTSPStream yeni bir RTSP stream yaradır.
func NewRTSPStream(cameraID, rtspURL string, bufferSize int) *RTSPStream {
	return &RTSPStream{
		URL:       rtspURL,
		CameraID:  cameraID,
		frameChan: make(chan *Frame, bufferSize),
		stopChan:  make(chan struct{}),
		isRunning: false,
	}
}

// Connect RTSP axınına qoşulmağa çalışır (retry logic ilə).
func (rs *RTSPStream) Connect(maxRetries int, retryInterval time.Duration) error {
	log.Printf("[RTSP] Camera %s: Connecting to %s", rs.CameraID, rs.URL)
	
	cURL := C.CString(rs.URL)
	defer C.free(unsafe.Pointer(cURL))

	for attempt := 1; attempt <= maxRetries; attempt++ {
		log.Printf("[RTSP] Camera %s: Connection attempt %d/%d", rs.CameraID, attempt, maxRetries)

		// Open Stream using FFmpeg C bindings
		rs.avCtx = C.open_rtsp_stream(cURL)
		if rs.avCtx != nil {
			rs.isRunning = true
			log.Printf("[RTSP] Camera %s: Connected successfully (res: %dx%d)", rs.CameraID, rs.avCtx.width, rs.avCtx.height)
			return nil
		}

		if attempt < maxRetries {
			time.Sleep(retryInterval)
		}
	}

	return errors.New("failed to connect to RTSP stream after retries")
}

// Start RTSP axınını oxumağa başlayır (ayrı goroutine-də).
func (rs *RTSPStream) Start() {
	if !rs.isRunning || rs.avCtx == nil {
		log.Printf("[RTSP] Stream %s is not connected", rs.CameraID)
		return
	}

	go rs.readFrames()
	log.Printf("[RTSP] Camera %s: Frame reading started", rs.CameraID)
}

// readFrames görüntü kadrlarını oxuyur və frameChan-a göndərir (Real FFmpeg).
func (rs *RTSPStream) readFrames() {
	defer func() {
		rs.isRunning = false
		if rs.avCtx != nil {
			C.close_rtsp_stream(rs.avCtx)
			rs.avCtx = nil
		}
		close(rs.frameChan)
		log.Printf("[RTSP] Camera %s: Frame reading stopped", rs.CameraID)
	}()

	width := int(rs.avCtx.width) 
	height := int(rs.avCtx.height)
	frameSize := width * height * 3

	for {
		select {
		case <-rs.stopChan:
			return
		default:
			// 1. Get Buffer
			var dataPtr unsafe.Pointer
			var dataSlice []byte

			if rs.Allocator != nil {
				// Use zero-copy allocator (C++ memory)
				dataPtr, dataSlice = rs.Allocator(frameSize)
			} else {
				// Fallback to Go memory
				dataSlice = make([]byte, frameSize)
				dataPtr = unsafe.Pointer(&dataSlice[0])
			}
			
			if dataPtr == nil {
				log.Printf("[RTSP] Camera %s: Failed to allocate buffer", rs.CameraID)
				time.Sleep(10 * time.Millisecond)
				continue
			}

			// 2. Read Frame directly into buffer
			ret := C.read_frame(rs.avCtx, (*C.uchar)(dataPtr), C.int(frameSize))
			
			if ret == 0 {
				// Success
				frame := &Frame{
					Data:      dataSlice,
					DataPtr:   dataPtr,
					Width:     width,
					Height:    height,
					Timestamp: time.Now(),
				}

				select {
				case rs.frameChan <- frame:
					// Sent
				case <-rs.stopChan:
					return
				}
			} else {
				// Error or EOF or Timeout
				// log.Printf("[RTSP] Camera %s: Read error or EOF (ret: %d)", rs.CameraID, ret)
				// Small sleep to avoid busy loop on error
				time.Sleep(10 * time.Millisecond)
			}
		}
	}
}

// GetFrameChannel frame channel-ını qaytarır.
func (rs *RTSPStream) GetFrameChannel() <-chan *Frame {
	return rs.frameChan
}

// Stop RTSP axınını dayandırır.
func (rs *RTSPStream) Stop() {
	if !rs.isRunning {
		return
	}
	// Avoid closing closed channel panic
	select {
	case <-rs.stopChan:
		return
	default:
		close(rs.stopChan)
	}
	log.Printf("[RTSP] Camera %s: Stop signal sent", rs.CameraID)
}
