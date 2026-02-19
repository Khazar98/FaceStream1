package main

/*
#cgo CFLAGS: -I../../pkg/cpp_processor
#cgo LDFLAGS: -L../../pkg/cpp_processor -lprocessor -lstdc++ -lcuda -lcudart -Wl,-rpath,../../pkg/cpp_processor


#include <stdlib.h>

// Define callback type matching C++: const char*
typedef void (*DetectionCallback)(const char* camera_id, const char* json_data);

// Declare C++ functions with correct signatures
int InitializeProcessor(const char* engine_path, DetectionCallback callback);
void ShutdownProcessor();
unsigned char* AllocateFrame(int size);
void FreeFrame(unsigned char* buffer);
void ProcessBatch(char** camera_ids, unsigned char** frame_data_array, int batch_size, int* widths, int* heights);
void ProcessFrame(const char* camera_id, unsigned char* frame_data, int width, int height);
void StopCameraStream(const char* camera_id);

// Forward declaration of Go exported function
extern void goDetectionCallback(char* camera_id, char* json_data);

// Wrapper function to match C++ signature and avoid direct Go function pointer casting issues
static void cDetectionCallbackWrapper(const char* camera_id, const char* json_data) {
    goDetectionCallback((char*)camera_id, (char*)json_data);
}

// Wrapper to cast Go function to C++ expected type
static int CallInitializeProcessor(char* path) {
    return InitializeProcessor(path, cDetectionCallbackWrapper);
}
*/
import "C"

import (
	"encoding/json"
	"log"
	"time"
	"unsafe"
	"video-analytics-system/internal/kafka"
)

// Global Kafka producer reference.
var globalKafkaProducer *kafka.Producer

// DetectionResult C++ tərəfindən gələn deteksiya nəticəsi.
type DetectionResult struct {
	CameraID          string          `json:"camera_id"`
	DetectionTime     string          `json:"detection_time"`
	ImageBase64       string          `json:"image_base64"`         // Tight face crop
	ImageBase64Padded string          `json:"image_base64_padded"`  // Head + shoulders
	QualityMetadata   QualityMetadata `json:"quality_metadata"`
	Confidence        float32         `json:"confidence"`
	AgeEstimate       int             `json:"age_estimate"`
	GenderEstimate    string          `json:"gender_estimate"`
	
	// Internal fields
	TrackingID int `json:"track_id"`
}

// QualityMetadata keyfiyyət məlumatları.
type QualityMetadata struct {
	Confidence          float32 `json:"confidence"`
	Sharpness           float32 `json:"sharpness"`
	Brightness          float32 `json:"brightness"`
	Contrast            float32 `json:"contrast"`
	Yaw                 float32 `json:"yaw"`
	Pitch               float32 `json:"pitch"`
	Roll                float32 `json:"roll"`
	BlurScore           float32 `json:"blur_score"`
	NoiseScore          float32 `json:"noise_score"`
	IlluminationQuality float32 `json:"illumination_quality"`
	OcclusionScore      float32 `json:"occlusion_score"`
	FaceSize            int     `json:"face_size"`
}
type QualityData struct {
	Sharpness  float32 `json:"sharpness"`
	Brightness float32 `json:"brightness"`
}

// AnalyticsData detectionMega topic-i üçün detallı məlumat.
type AnalyticsData struct {
	CameraID    string  `json:"camera_id"`
	FaceWidth   int     `json:"face_width"`
	FaceHeight  int     `json:"face_height"`
	Sharpness   float32 `json:"sharpness"`
	Brightness  float32 `json:"brightness"`
	ImageBase64 string  `json:"image_base64"`
}

// InitializeAIProcessor C++ nüvəsini başladır (Go tərəfindən çağırılır).
func InitializeAIProcessor(enginePath string, producer *kafka.Producer) error {
	globalKafkaProducer = producer

	// C callback-i hazırla.
	cEnginePathPtr := C.CString(enginePath)
	defer C.free(unsafe.Pointer(cEnginePathPtr))

	// C++ InitializeProcessor-ü çağır wrapper vasitəsilə
	ret := C.CallInitializeProcessor(cEnginePathPtr)

	if ret < 0 {
		return &Error{Code: int(ret), Message: "Failed to initialize AI processor"}
	}

	log.Printf("[CGO] AI Processor initialized with engine: %s", enginePath)
	return nil
}

// ShutdownAIProcessor C++ nüvəsini dayandırır.
func ShutdownAIProcessor() {
	log.Println("[CGO] Shutting down AI Processor...")
	C.ShutdownProcessor()
}

// AllocateFrame allocates a frame buffer in C++ memory.
func AllocateFrame(size int) unsafe.Pointer {
	return unsafe.Pointer(C.AllocateFrame(C.int(size)))
}

// FreeFrame frees a frame buffer in C++ memory.
func FreeFrame(ptr unsafe.Pointer) {
	C.FreeFrame((*C.uchar)(ptr))
}

// ProcessBatch sends a batch of frames to C++.
// Updated for Batch 128 Support with Multi-Camera IDs and Mixed Resolutions
func ProcessBatch(cameraIDs []string, framePtrs []unsafe.Pointer, widths []int, heights []int) {
	if len(framePtrs) == 0 || len(cameraIDs) != len(framePtrs) || len(widths) != len(framePtrs) || len(heights) != len(framePtrs) {
		return
	}

	// Convert Go slice of strings to C array of strings (char**)
	cCameraIDs := make([]*C.char, len(cameraIDs))
	for i, id := range cameraIDs {
		cStr := C.CString(id)
		defer C.free(unsafe.Pointer(cStr))
		cCameraIDs[i] = cStr
	}

	// Convert Go slice of pointers to C array of pointers (unsigned char**)
	cPtrs := make([]*C.uchar, len(framePtrs))
	for i, ptr := range framePtrs {
		cPtrs[i] = (*C.uchar)(ptr)
	}

    // Convert Go slice of ints to C array of ints
    cWidths := make([]C.int, len(widths))
    cHeights := make([]C.int, len(heights))
    for i := range widths {
        cWidths[i] = C.int(widths[i])
        cHeights[i] = C.int(heights[i])
    }

	C.ProcessBatch(
		&cCameraIDs[0], // Address of the first string pointer (char**)
		&cPtrs[0],      // Address of the first frame pointer (unsigned char**)
		C.int(len(framePtrs)),
		&cWidths[0],    // Address of the first width
		&cHeights[0],   // Address of the first height
	)
}

// ProcessFrame kadrı C++-a göndərir (Deprecated wrapper).
func ProcessFrame(cameraID string, frameData []byte, width, height int) {
	if len(frameData) == 0 {
		return
	}

	cCameraID := C.CString(cameraID)
	defer C.free(unsafe.Pointer(cCameraID))

	// Frame data-nı unsafe.Pointer-ə çevirik.
	framePtr := (*C.uchar)(unsafe.Pointer(&frameData[0]))

	C.ProcessFrame(cCameraID, framePtr, C.int(width), C.int(height))
}

// ProcessFramePtr kadrı C++-a göndərir (pointer versiya)
func ProcessFramePtr(cameraID string, framePtr unsafe.Pointer, width, height int) {
	if framePtr == nil {
		return
	}

	cCameraID := C.CString(cameraID)
	defer C.free(unsafe.Pointer(cCameraID))

	C.ProcessFrame(cCameraID, (*C.uchar)(framePtr), C.int(width), C.int(height))
}

// StopAIStream müəyyən bir kamera axınını dayandırır.
func StopAIStream(cameraID string) {
	cCameraID := C.CString(cameraID)
	defer C.free(unsafe.Pointer(cCameraID))

	C.StopCameraStream(cCameraID)
	log.Printf("[CGO] Stream stopped for camera: %s", cameraID)
}

// goDetectionCallback Go-dan C++ tərəfindən çağırılan callback funksiyası.
//
//export goDetectionCallback
func goDetectionCallback(cameraIDPtr *C.char, jsonDataPtr *C.char) {
	if cameraIDPtr == nil || jsonDataPtr == nil {
		log.Println("[CGO] Received nil pointer in callback")
		return
	}

	cameraID := C.GoString(cameraIDPtr)
	jsonData := C.GoString(jsonDataPtr)

	// JSON məlumatını parse edin.
	var result DetectionResult
	if err := json.Unmarshal([]byte(jsonData), &result); err != nil {
		log.Printf("[CGO] Failed to unmarshal JSON from C++: %v", err)
		return
	}

	// Zaman və camera ID-ni yoxla.
	result.CameraID = cameraID
	if result.DetectionTime == "" {
		result.DetectionTime = time.Now().UTC().Format(time.RFC3339)
	}

	// Nəticəni Kafka-ya göndər (gate-metric-detections topic).
	sendToKafka(result)

	// Log output.
	log.Printf("[DETECTION] Camera: %s, TrackingID: %d, Confidence: %.2f, Sharpness: %.2f",
		cameraID, result.TrackingID, result.Confidence, result.QualityMetadata.Sharpness)
}

// sendToKafka deteksiya nəticəsini Kafka-ya göndərir.
func sendToKafka(result DetectionResult) {
	if globalKafkaProducer == nil {
		log.Println("[KAFKA] Producer is not initialized")
		return
	}

	// Confidence filter: 0.72-dən aşağı göndərmə
	if result.Confidence < 0.72 {
		log.Printf("[KAFKA] Skipped low confidence: %.2f (TrackingID=%d)", result.Confidence, result.TrackingID)
		return
	}

	// detectiontopic topic-ə göndər
	jsonPayload, err := json.Marshal(result)
	if err != nil {
		log.Printf("[KAFKA] Failed to marshal detection result: %v", err)
		return
	}

	if err := globalKafkaProducer.SendMessage("detectiontopic", string(jsonPayload)); err != nil {
		log.Printf("[KAFKA] Failed to send message: %v", err)
	} else {
		log.Printf("[KAFKA] ✓ Sent: TrackingID=%d, Confidence=%.2f, Sharpness=%.2f", 
			result.TrackingID, result.Confidence, result.QualityMetadata.Sharpness)
	}
}

// Error custom xəta tipi.
type Error struct {
	Code    int
	Message string
}

func (e *Error) Error() string {
	return e.Message
}
