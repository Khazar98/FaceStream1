#include "processor.h"
#include "tracker.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>
#include <fstream>

// OpenCV Headers
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// TensorRT & CUDA Headers
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// Logger for TensorRT
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors to avoid spam
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << " at line " << __LINE__ << std::endl; \
            return; \
        } \
    } while (0)

// Standard RFC 4648 base64 encoding
std::string simple_base64_encode(const std::vector<unsigned char>& input) {
	static const char* base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string ret;
	size_t i = 0;
	unsigned char char_array_3[3];
	unsigned char char_array_4[4];
	size_t in_len = input.size();

	while (in_len--) {
		char_array_3[i++] = *(input.data() + (input.size() - in_len - 1));
		if (i == 3) {
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;

			for (i = 0; i < 4; i++)
				ret += base64_chars[char_array_4[i]];
			i = 0;
		}
	}

	if (i) {
		for (size_t j = i; j < 3; j++)
			char_array_3[j] = '\0';

		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

		for (size_t j = 0; j < i + 1; j++)
			ret += base64_chars[char_array_4[j]];

		while (i++ < 3)
			ret += '=';
	}

	return ret;
}

// Frame Buffer Structure
struct BestShotFrame {
	std::string camera_id;
	int track_id;
	std::vector<unsigned char> frame_data;  // Tight crop
	std::vector<unsigned char> padded_frame_data;  // Head+shoulders crop
	int width;
	int height;
	int padded_width;
	int padded_height;
	float quality_score;
	std::chrono::system_clock::time_point timestamp;
	float confidence;
	float sharpness;
	float brightness;
};

// Camera Stream Context
struct CameraContext {
	std::string camera_id;
	std::map<int, BestShotFrame> best_shots;
	std::map<int, std::chrono::system_clock::time_point> track_last_seen;
    std::map<int, std::chrono::system_clock::time_point> track_first_seen;
    std::map<int, int> track_detection_count;
    std::map<int, std::chrono::system_clock::time_point> track_last_sent; // Son gönderilme vaxti (track bazlı)
    std::chrono::system_clock::time_point camera_last_sent; // [NEW] Kamera üzrə son gönderilme
	std::mutex mtx;
    std::unique_ptr<BoTSORT> tracker;
    
    CameraContext() : camera_last_sent(std::chrono::system_clock::now() - std::chrono::seconds(30)) {
        // BoTSORT default: track_high=0.6, track_low=0.1, new_track=0.65, match=0.60, buffer=60 (2s)
        tracker = std::make_unique<BoTSORT>(0.6f, 0.1f, 0.65f, 60, 0.60f);
    }
};

// Global state
static DetectionCallback detection_callback_ = nullptr;
static std::map<std::string, std::unique_ptr<CameraContext>> camera_contexts;
static std::mutex contexts_mtx;
static std::mutex inference_mtx; 
static std::thread timeout_thread_;
static bool shutdown_flag_ = false;

// Forward declarations
void timeout_worker_thread();
std::string frame_to_base64(const std::vector<unsigned char>& data, int width, int height);
float calculate_sharpness_sobel(const std::vector<unsigned char>& frame_data, int width, int height);
float calculate_brightness(const std::vector<unsigned char>& frame_data);
std::vector<unsigned char> crop_face_region(const unsigned char* frame_data, int frame_width, int frame_height, 
                                             float x, float y, float w, float h, int* out_width, int* out_height, float padding_factor);

// TensorRT Global State
static TRTLogger gLogger;
static nvinfer1::IRuntime* trt_runtime = nullptr;
static nvinfer1::ICudaEngine* trt_engine = nullptr;
static nvinfer1::IExecutionContext* trt_context = nullptr;
static cudaStream_t trt_stream = nullptr;

// Tensor Names
static std::string inputName;
static std::string outputName;

// Buffer pointers
static void* buffers[2]; // 0: Input, 1: Output
static size_t inputSize = 0;
static size_t outputSize = 0;

// Initialize Processor (TensorRT 10)
// Initialize Processor (TensorRT 10)
int InitializeProcessor(const char* model_path, DetectionCallback callback) {
    std::cerr << "[C++] InitializeProcessor ENTRY" << std::endl;
    detection_callback_ = callback;
    if (!detection_callback_) return -1;

    std::cout << "[C++] Initializing TensorRT Processor (TRT 10)..." << std::endl;

    // Create runtime
    trt_runtime = nvinfer1::createInferRuntime(gLogger);
    if (!trt_runtime) {
        std::cerr << "[C++] Failed to create TensorRT Runtime." << std::endl;
        return -1;
    }

    // Load engine file
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "[C++] Error reading engine file: " << model_path << std::endl;
        return -1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
         std::cerr << "[C++] Error loading engine file content." << std::endl;
         return -1;
    }

    // Deserialize engine
    char* engine_data = buffer.data();
    size_t engine_size = size;

    // Check for TRT Magic and skip header if necessary
    // Expected Layout: "ftrt" (Little Endian: 0x66, 0x74, 0x72, 0x74)
    // We check if the file starts with it. If not, we scan.
    
    unsigned char magic[] = {0x66, 0x74, 0x72, 0x74};
    bool magic_at_start = (size > 4 && 
                           (unsigned char)buffer[0] == magic[0] && 
                           (unsigned char)buffer[1] == magic[1] && 
                           (unsigned char)buffer[2] == magic[2] && 
                           (unsigned char)buffer[3] == magic[3]);

    if (!magic_at_start) {
        std::cout << "[C++] TRT Magic not at start. Scanning for magic tag 'ftrt'..." << std::endl;
        size_t offset = 0;
        bool found = false;
        size_t search_limit = (size < 10000) ? size : 10000;
        
        // Scan byte by byte
        for (size_t i = 0; i < search_limit - 4; i++) {
            if ((unsigned char)buffer[i] == magic[0] && 
                (unsigned char)buffer[i+1] == magic[1] && 
                (unsigned char)buffer[i+2] == magic[2] && 
                (unsigned char)buffer[i+3] == magic[3]) {
                offset = i;
                found = true;
                break;
            }
        }
        
        if (found) {
            std::cout << "[C++] Found TRT magic at offset " << offset << ". Skipping header." << std::endl;
            engine_data += offset;
            engine_size -= offset;
        } else {
             std::cerr << "[C++] CRITICAL WARNING: TRT Magic 'ftrt' not found in first 10KB!" << std::endl;
             // We continue anyway, letting deserialize fail with a clear error if it must
        }
    } else {
         std::cout << "[C++] TRT Magic found at start of file." << std::endl;
    }

    trt_engine = trt_runtime->deserializeCudaEngine(engine_data, engine_size);
    if (!trt_engine) {
        std::cerr << "[C++] Failed to deserialize CUDA engine." << std::endl;
        return -1;
    }

    // Create execution context
    trt_context = trt_engine->createExecutionContext();
    if (!trt_context) {
        std::cerr << "[C++] Failed to create execution context." << std::endl;
        return -1;
    }

    // Create CUDA stream
    if (cudaStreamCreate(&trt_stream) != cudaSuccess) {
        std::cerr << "[C++] Failed to create CUDA stream." << std::endl;
        return -1;
    }

    // Find input/output tensor names
    int nbIOTensors = trt_engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = trt_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = trt_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            std::cout << "[C++] Found Input Tensor: " << inputName << std::endl;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName = name; // Assuming only one output
            std::cout << "[C++] Found Output Tensor: " << outputName << std::endl;
            // DEBUG: Log output shape
            nvinfer1::Dims dims = trt_engine->getTensorShape(name);
            std::cout << "[C++] Output Shape: [";
            for (int d = 0; d < dims.nbDims; d++) {
                std::cout << dims.d[d];
                if (d < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    if (inputName.empty() || outputName.empty()) {
        std::cerr << "[C++] Failed to find input/output tensors." << std::endl;
        return -1;
    }

    // Setup buffers (Dynamic shapes handled in ProcessBatch)
    // Allocating max buffers
    int max_batch = 128; // Default max
    inputSize = max_batch * 3 * 640 * 640 * sizeof(float);
    // Output size: previously 5 * 8400, verify if it's diff layout
    // Assuming output is [batch, 4+conf, 8400] -> [B, 5, 8400] or similar.
    // If output is dynamic, we allocate max possible.
    outputSize = max_batch * 5 * 8400 * sizeof(float);

    if (cudaMalloc(&buffers[0], inputSize) != cudaSuccess) {
         std::cerr << "[C++] Failed to allocate CUDA input buffer." << std::endl;
         return -1;
    }
    if (cudaMalloc(&buffers[1], outputSize) != cudaSuccess) {
         std::cerr << "[C++] Failed to allocate CUDA output buffer." << std::endl;
         return -1;
    }
    
    // Set tensor addresses once? No, dynamic shapes might require setting shape, then address?
    // Actually, setTensorAddress can be set anytime before enqueue.
    if (!trt_context->setTensorAddress(inputName.c_str(), buffers[0])) {
         std::cerr << "[C++] Failed to set input tensor address." << std::endl;
    }
    if (!trt_context->setTensorAddress(outputName.c_str(), buffers[1])) {
         std::cerr << "[C++] Failed to set output tensor address." << std::endl;
    }

    std::cout << "[C++] TensorRT initialized successfully. Max Batch: " << max_batch << std::endl;

    shutdown_flag_ = false;
    timeout_thread_ = std::thread(timeout_worker_thread);
    return 0;
}


// Timeout worker
void timeout_worker_thread() {
	const int TIMEOUT_MS = 2000; // 2 seconds (max_time_lost)
	while (!shutdown_flag_) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms check interval
		
        std::vector<BestShotFrame> pending_best_shots;
        {
            std::lock_guard<std::mutex> lock(contexts_mtx);
            auto now = std::chrono::system_clock::now();
            
            for (auto& [camera_id, context_ptr] : camera_contexts) {
                CameraContext& context = *context_ptr;
                std::lock_guard<std::mutex> ctx_lock(context.mtx);
                std::vector<int> timed_out_tracks;
                
                for (auto& [track_id, last_seen] : context.track_last_seen) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_seen);
                    if (duration.count() >= TIMEOUT_MS) {
                        timed_out_tracks.push_back(track_id);
                    }
                }
                
                for (int track_id : timed_out_tracks) {
                    std::cerr << "[C++-TIMEOUT] Track " << track_id << " timed out, best_shot_exists=" 
                              << (context.best_shots.find(track_id) != context.best_shots.end()) << std::endl;
                    if (context.best_shots.find(track_id) != context.best_shots.end()) {
                        // Send best shot when track times out (2 seconds)
                        pending_best_shots.push_back(std::move(context.best_shots[track_id]));
                        context.best_shots.erase(track_id);
                    }
                    // Clean up track data
                    context.track_last_seen.erase(track_id);
                    context.track_first_seen.erase(track_id); 
                    context.track_detection_count.erase(track_id);
                    context.track_last_sent.erase(track_id);
                }
            }
        } 
        
        for (const auto& best : pending_best_shots) {
            std::string image_b64 = frame_to_base64(best.frame_data, best.width, best.height);
            std::string image_b64_padded = frame_to_base64(best.padded_frame_data, best.padded_width, best.padded_height);
            
                              // Base64 encoding verified
            
            // [FIX] Proper JSON construction with explicit buffer sizing
            std::string json_payload = "{";
            json_payload += "\"camera_id\":\"" + best.camera_id + "\",";
            json_payload += "\"track_id\":" + std::to_string(best.track_id) + ",";
            json_payload += "\"detection_time\":\"2023-10-27T10:00:00Z\",";
            json_payload += "\"image_base64\":\"" + image_b64 + "\",";
            json_payload += "\"image_base64_padded\":\"" + image_b64_padded + "\",";
            json_payload += "\"quality_metadata\":{";
            json_payload += "\"confidence\":" + std::to_string(best.confidence) + ",";
            json_payload += "\"sharpness\":" + std::to_string(best.sharpness) + ",";
            json_payload += "\"brightness\":" + std::to_string(best.brightness) + ",";
            json_payload += "\"contrast\":0,\"yaw\":0,\"pitch\":0,\"roll\":0,";
            json_payload += "\"blur_score\":0,\"noise_score\":0,\"illumination_quality\":0,";
            json_payload += "\"occlusion_score\":0,\"face_size\":0";
            json_payload += "},";
            json_payload += "\"confidence\":" + std::to_string(best.confidence) + ",";
            json_payload += "\"age_estimate\":0,\"gender_estimate\":\"unknown\"";
            json_payload += "}";
            
                              // JSON payload ready
            
            if (detection_callback_) {
                detection_callback_(const_cast<char*>(best.camera_id.c_str()), const_cast<char*>(json_payload.c_str()));
            }
        }
	}
}

float calculate_sharpness_sobel(const std::vector<unsigned char>& frame_data, int width, int height) {
    if (width < 3 || height < 3) return 0.0f;
    float sum_gradients = 0.0f;
    int pixel_count = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = (y * width + x) * 3;
            int gx = -frame_data[idx - width*3] + frame_data[idx + width*3];
            int gy = -frame_data[idx - 3] + frame_data[idx + 3];
            float gradient = std::sqrt(gx*gx + gy*gy);
            sum_gradients += gradient;
            pixel_count++;
        }
    }
    return (pixel_count > 0) ? std::min(100.0f, ((sum_gradients / pixel_count) / 5.0f)) : 0.0f;
}

float calculate_brightness(const std::vector<unsigned char>& frame_data) {
    if (frame_data.empty()) return 0.0f;
    long long sum = 0;
    for (unsigned char val : frame_data) sum += val;
    return ((float)sum / frame_data.size()) / 255.0f * 100.0f;
}

std::string frame_to_base64(const std::vector<unsigned char>& data, int width, int height) {
    if (data.empty() || width <= 0 || height <= 0) return "";
    cv::Mat img(height, width, CV_8UC3, (void*)data.data());
    std::vector<unsigned char> jpeg_buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
    if (!cv::imencode(".jpg", img, jpeg_buffer, params)) return "";
    return simple_base64_encode(jpeg_buffer);
}

std::vector<unsigned char> crop_face_region(const unsigned char* frame_data, int frame_width, int frame_height,
                                             float x, float y, float w, float h, int* out_width, int* out_height, float padding_factor) {
    float pad_w = w * padding_factor;
    float pad_h = h * padding_factor;
    
    int x1 = std::max(0, (int)(x - pad_w));
    int y1 = std::max(0, (int)(y - pad_h));
    int x2 = std::min(frame_width, (int)(x + w + pad_w));
    int y2 = std::min(frame_height, (int)(y + h + pad_h));
    
    int crop_w = x2 - x1;
    int crop_h = y2 - y1;
    
    if (crop_w <= 0 || crop_h <= 0) {
        *out_width = 0;
        *out_height = 0;
        return std::vector<unsigned char>();
    }
    
    *out_width = crop_w;
    *out_height = crop_h;
    
    std::vector<unsigned char> cropped(crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++) {
        for (int x = 0; x < crop_w; x++) {
            int src_idx = ((y1 + y) * frame_width + (x1 + x)) * 3;
            int dst_idx = (y * crop_w + x) * 3;
            cropped[dst_idx + 0] = frame_data[src_idx + 0]; // B
            cropped[dst_idx + 1] = frame_data[src_idx + 1]; // G
            cropped[dst_idx + 2] = frame_data[src_idx + 2]; // R
        }
    }
    
    return cropped;
}

unsigned char* AllocateFrame(int size) {
    return new unsigned char[size];
}

void FreeFrame(unsigned char* buffer) {
    if (buffer) delete[] buffer;
}

extern "C" {
    void ProcessBatch(char** camera_ids, unsigned char** frame_data_array, int batch_size, int* widths, int* heights) {
         std::lock_guard<std::mutex> lock(inference_mtx); // [FIX] Protect entire batch processing including static buffer access
         
         if (!camera_ids || !frame_data_array || !widths || !heights || batch_size <= 0) return;
         if (batch_size > 128) batch_size = 128;

         static std::vector<float> inputBlob(128 * 3 * 640 * 640);
         
         // [CRITICAL FIX] Clear ENTIRE input blob to prevent stale data from previous batches
         std::fill(inputBlob.begin(), inputBlob.end(), 0.0f); 
         
         struct FrameMeta {
             std::string cam_id;
             float scale;
             int dw, dh;
             int original_w, original_h;
         };
         std::vector<FrameMeta> batch_meta(batch_size);
         
         for (int i = 0; i < batch_size; i++) {
             unsigned char* frame_data = frame_data_array[i];
             char* cam_id_cstr = camera_ids[i];
             int w = widths[i];
             int h = heights[i];

             if (!frame_data || !cam_id_cstr || w <= 0 || h <= 0) continue;
             
             float scale = std::min(640.0f / w, 640.0f / h);
             int new_w = std::round(w * scale);
             int new_h = std::round(h * scale);
             int dw = (640 - new_w) / 2;
             int dh = (640 - new_h) / 2;
             
             batch_meta[i] = {std::string(cam_id_cstr), scale, dw, dh, w, h};
             
             cv::Mat frame(h, w, CV_8UC3, frame_data);
             cv::Mat resized_frame;
             if (w != new_w || h != new_h) {
                 cv::resize(frame, resized_frame, cv::Size(new_w, new_h));
             } else {
                 resized_frame = frame;
             }
             
             cv::Mat padded_frame;
             cv::copyMakeBorder(resized_frame, padded_frame, dh, 640-new_h-dh, dw, 640-new_w-dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
             
             size_t offset = i * 3 * 640 * 640;
             
             // DEBUG: Save first frame of first batch for verification
             static bool debug_dumped = false;
             bool should_dump = !debug_dumped && i == 0;
             
             for(int r=0; r<640; ++r) {
                 for(int c=0; c<640; ++c) {
                     cv::Vec3b pixel = padded_frame.at<cv::Vec3b>(r, c);
                     // [FIX] Normalize by 255.0 to match Python/Ultralytics preprocessing
                     inputBlob[offset + r*640 + c] = (float)pixel[2] / 255.0f;           // R
                     inputBlob[offset + 640*640 + r*640 + c] = (float)pixel[1] / 255.0f; // G
                     inputBlob[offset + 2*640*640 + r*640 + c] = (float)pixel[0] / 255.0f; // B
                 }
             }
             
             // DEBUG: Dump after filling tensor
             if (should_dump) {
                 cv::imwrite("/home/admiral/Khazar/CProjects/FaceStream1/debug_cpp_padded.jpg", padded_frame);
                 // Dump raw tensor (first 10 values of each channel)
                 std::ofstream tensor_file("/home/admiral/Khazar/CProjects/FaceStream1/debug_cpp_tensor.txt");
                 tensor_file << "Scale: " << scale << " dw: " << dw << " dh: " << dh << "\n";
                 tensor_file << "First 10 values of each channel:\n";
                 for(int ch=0; ch<3; ++ch) {
                     tensor_file << "Channel " << ch << ": ";
                     for(int idx=0; idx<10; ++idx) {
                         tensor_file << inputBlob[offset + ch*640*640 + idx] << " ";
                     }
                     tensor_file << "\n";
                 }
                 // Sample a few pixels from center of image
                 int center_r = 320, center_c = 320;
                 tensor_file << "\nSample pixels from center (320,320):\n";
                 tensor_file << "R: " << inputBlob[offset + center_r*640 + center_c] << "\n";
                 tensor_file << "G: " << inputBlob[offset + 640*640 + center_r*640 + center_c] << "\n";
                 tensor_file << "B: " << inputBlob[offset + 2*640*640 + center_r*640 + center_c] << "\n";
                 tensor_file.close();
                 debug_dumped = true;
                 std::cout << "[C++] DEBUG: Saved debug_cpp_padded.jpg and debug_cpp_tensor.txt" << std::endl;
             }
         }
         
         constexpr int MAX_DET = 8400;
         constexpr int NUM_ATTR = 5;
         
         if (trt_context) {
             // Set Input Dimensions for Dynamic Shapes (TRT 10)
             nvinfer1::Dims4 inputDims(batch_size, 3, 640, 640);
             if (!trt_context->setInputShape(inputName.c_str(), inputDims)) {
                 std::cerr << "[C++] Warning: Failed to set input shape." << std::endl;
             }

             size_t current_input_size = batch_size * 3 * 640 * 640 * sizeof(float);
             size_t current_output_size = batch_size * NUM_ATTR * MAX_DET * sizeof(float); // Need to verify if output shape changes with batch


             
             // DEBUG: Check input blob min/max values
             float input_max = 0.0f;
             float input_min = 1.0f;
             for(size_t idx=0; idx < current_input_size / sizeof(float); idx++) {
                 if (inputBlob[idx] > input_max) input_max = inputBlob[idx];
                 if (inputBlob[idx] < input_min) input_min = inputBlob[idx];
             }
             static int input_debug_cnt = 0;
             if (input_debug_cnt++ < 50) {
                 std::cerr << "[C++-INPUT] Batch=" << batch_size << " min=" << input_min 
                           << " max=" << input_max << std::endl;
             }
             
             // Copy Input to GPU
             CHECK_CUDA(cudaMemcpyAsync(buffers[0], inputBlob.data(), current_input_size, cudaMemcpyHostToDevice, trt_stream));
             
             // [CRITICAL FIX] Synchronize to ensure input is fully copied before inference
             CHECK_CUDA(cudaStreamSynchronize(trt_stream));
             
             // Run Inference (enqueueV3)
             if (!trt_context->enqueueV3(trt_stream)) {
                 std::cerr << "[C++] TensorRT Inference Failed (enqueueV3)." << std::endl;
                 return;
             }
             
             // Copy Output to CPU
             static std::vector<float> outputHost(128 * NUM_ATTR * MAX_DET);
             CHECK_CUDA(cudaMemcpyAsync(outputHost.data(), buffers[1], current_output_size, cudaMemcpyDeviceToHost, trt_stream));
             CHECK_CUDA(cudaStreamSynchronize(trt_stream));

             float* output_data = outputHost.data();
             
             // DEBUG: Check output data sanity
             static int out_debug = 0;
             if (out_debug++ < 50) {
                 float max_conf = 0;
                 for (int b = 0; b < MAX_DET; b++) {
                     float conf = output_data[0 + 4 * MAX_DET + b]; // First frame, confidence channel
                     if (conf > max_conf) max_conf = conf;
                 }
                 std::cerr << "[C++-OUTPUT] Batch=" << batch_size << " max_conf=" << max_conf << std::endl;
             }
                
                for (int i = 0; i < batch_size; i++) {
                     if (batch_meta[i].cam_id.empty()) continue;
                     
                     std::string cam_id = batch_meta[i].cam_id;
                     int orig_w = batch_meta[i].original_w;
                     int orig_h = batch_meta[i].original_h;
                     
                     {
                        std::lock_guard<std::mutex> lock(contexts_mtx);
                        if (camera_contexts.find(cam_id) == camera_contexts.end()) {
                            camera_contexts[cam_id] = std::make_unique<CameraContext>();
                            camera_contexts[cam_id]->camera_id = cam_id;
                        }
                     }
                     
                     CameraContext& contextState = *camera_contexts[cam_id];
                     std::lock_guard<std::mutex> lock(contextState.mtx);
                     
                     // DEBUG: Log context and frame info
                     static int ctx_debug = 0;
                     if (ctx_debug++ < 100) {
                         std::cerr << "[C++-CTX] cam=" << cam_id << " frame=" << i 
                                   << " frame_ptr=" << (void*)frame_data_array[i]
                                   << " ctx_tracker=" << (void*)contextState.tracker.get()
                                   << " orig_size=" << orig_w << "x" << orig_h << std::endl;
                     }
    
                     std::vector<std::vector<float>> current_detections;
                     
                     // YOLO Output layout: [Batch, Attributes, Detections] -> [B, 5, 8400]
                     // Offset for batch i = i * (5 * 8400)
                     size_t frame_offset = i * (NUM_ATTR * MAX_DET);

                     // Wait, typical YOLO output is [batch, 4+cls, 8400]. 
                     // e.g. [1, 5, 8400] for 1 class.
                     // The data is contiguous. 
                     // x = data[frame_offset + 0*8400 + k]
                     // y = data[frame_offset + 1*8400 + k]
                     // w = data[frame_offset + 2*8400 + k]
                     // h = data[frame_offset + 3*8400 + k]
                     // conf = data[frame_offset + 4*8400 + k]
                     
                     
                     // DEBUG: Check output tensor max values
                     static int cam_output_debug = 0;
                     bool should_log_output = cam_output_debug++ < 100;
                     if (should_log_output) {
                         float max_conf = 0.0f;
                         int max_idx = -1;
                         int high_conf_count = 0;
                         for (int b = 0; b < MAX_DET; b++) {
                             float conf = output_data[frame_offset + 4 * MAX_DET + b];
                             if (conf > max_conf) {
                                 max_conf = conf;
                                 max_idx = b;
                             }
                             if (conf > 0.5f) high_conf_count++;
                         }
                         std::cerr << "[C++-OUT] Max conf: " << max_conf << " at index " << max_idx 
                                   << " high_conf(>0.5): " << high_conf_count << std::endl;
                     }

                     // [FIX] Raised threshold to reduce false positives (was 0.50 -> 0.60 -> 0.70)
                     // [FIX] Lowered threshold to match Ultralytics default (was 0.70)
                     float conf_threshold = 0.45f; 
                     int det_count = 0;
                     
                     // DEBUG: Log first few detections for each camera
                     static int cam_debug_count = 0;
                     bool should_log_dets = cam_debug_count++ < 100;
                     
                     for (int b = 0; b < MAX_DET; b++) {
                         float x = output_data[frame_offset + 0 * MAX_DET + b];
                         float y = output_data[frame_offset + 1 * MAX_DET + b];
                         float w = output_data[frame_offset + 2 * MAX_DET + b];
                         float h = output_data[frame_offset + 3 * MAX_DET + b];
                         float confidence = output_data[frame_offset + 4 * MAX_DET + b];
                         
                         if (confidence >= conf_threshold) {
                             det_count++;
                             if (should_log_dets && det_count <= 10) {
                                 std::cerr << "[C++-DET] " << det_count << " conf=" << confidence 
                                           << " xywh=[" << x << "," << y << "," << w << "," << h << "]" << std::endl;
                             }
                             float scale = batch_meta[i].scale;
                             int dw = batch_meta[i].dw;
                             int dh = batch_meta[i].dh;
                             
                             float x_center = (x - dw) / scale;
                             float y_center = (y - dh) / scale;
                             float w_original = w / scale;
                             float h_original = h / scale;
                             
                             // [DEBUG TRAP] Save image for high-confidence floor detections (>0.60)
                             if (confidence > 0.60f && (y_center / orig_h) > 0.75f) {
                                  static int fp_dump_count = 0;
                                  if (fp_dump_count < 5) {
                                      std::string filename = "/home/admiral/Khazar/CProjects/FaceStream1/debug_fp_" + std::to_string(fp_dump_count) + ".jpg";
                                      
                                      // Reconstruct frame to save it
                                      cv::Mat raw_frame(orig_h, orig_w, CV_8UC3, frame_data_array[i]);
                                      cv::imwrite(filename, raw_frame);
                                      
                                      std::cerr << "[C++-DEBUG] SAVED FALSE POSITIVE INPUT: " << filename 
                                                << " conf=" << confidence << " y=" << y_center << std::endl;
                                      fp_dump_count++;
                                  }
                             }
                             
                             float x1 = std::max(0.0f, std::min(x_center - w_original/2.0f, (float)orig_w - 1));
                             float y1 = std::max(0.0f, std::min(y_center - h_original/2.0f, (float)orig_h - 1));
                             float box_w_val = std::max(1.0f, std::min(w_original, (float)orig_w - x1));
                             float box_h_val = std::max(1.0f, std::min(h_original, (float)orig_h - y1));
                             
                             // DEBUG: Log detection coordinates before adding
                             static int det_coord_debug = 0;
                             if (det_coord_debug++ < 200) {
                                 std::cerr << "[C++-DET-RAW] conf=" << confidence 
                                           << " x=" << x << " y=" << y << " w=" << w << " h=" << h
                                           << " | scaled: x1=" << x1 << " y1=" << y1 
                                           << " box_w=" << box_w_val << " box_h=" << box_h_val << std::endl;
                             }
                             
                             current_detections.push_back({0.0f, confidence, x1, y1, box_w_val, box_h_val});
                         }
                     }
                     
                     // DEBUG: Log detection count
                     static int det_count_debug = 0;
                     if (det_count_debug++ < 100) {
                         std::cerr << "[C++-DETS] " << cam_id << " Raw detections: " << current_detections.size() << std::endl;
                     }
                     
                     if (current_detections.size() > 1) {
                         std::sort(current_detections.begin(), current_detections.end(), 
                             [](const std::vector<float>& a, const std::vector<float>& b) { return a[1] > b[1]; });
                         
                         std::vector<std::vector<float>> nms_result;
                         std::vector<bool> suppressed(current_detections.size(), false);
                         for(size_t j=0; j<current_detections.size(); ++j) {
                             if(suppressed[j]) continue;
                             nms_result.push_back(current_detections[j]);
                             for(size_t k=j+1; k<current_detections.size(); ++k) {
                                 if(suppressed[k]) continue;
                                 float inter_x1 = std::max(current_detections[j][2], current_detections[k][2]);
                                 float inter_y1 = std::max(current_detections[j][3], current_detections[k][3]);
                                 float inter_x2 = std::min(current_detections[j][2]+current_detections[j][4], current_detections[k][2]+current_detections[k][4]);
                                 float inter_y2 = std::min(current_detections[j][3]+current_detections[j][5], current_detections[k][3]+current_detections[k][5]);
                                 float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                                 float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                                 float inter_area = inter_w * inter_h;
                                 float union_area = (current_detections[j][4]*current_detections[j][5]) + (current_detections[k][4]*current_detections[k][5]) - inter_area;
                                 // NMS IoU threshold
                              if (union_area > 0 && inter_area / union_area > 0.45f) suppressed[k] = true;
                             }
                         }
                         current_detections = nms_result;
                     }
                     // DEBUG: Log after NMS (always log for debugging)
                     static int nms_debug_count = 0;
                     if (nms_debug_count++ < 100) {
                         std::cerr << "[C++-NMS] " << cam_id << " After NMS: " << current_detections.size() << std::endl;
                     }
                     
                     if (contextState.tracker) {
                          std::vector<Track> tracks = contextState.tracker->update(current_detections);
                          for (const auto& t : tracks) {
                              int track_id = t.track_id;
                              float confidence = t.score;
                              
                              // Note: BoTSORT buffer=1000, we control timeout ourselves (2 seconds)
                              
                              float x1 = t.bbox[0]; float y1 = t.bbox[1];
                              float box_w = t.bbox[2]; float box_h = t.bbox[3];
                              
                              // Frame validation passed
                              
                              int crop_w, crop_h, padded_w, padded_h;
                              unsigned char* frame_raw = frame_data_array[i];
                              std::vector<unsigned char> face_crop = crop_face_region(frame_raw, orig_w, orig_h, x1, y1, box_w, box_h, &crop_w, &crop_h, 0.0f);
                              std::vector<unsigned char> face_crop_padded = crop_face_region(frame_raw, orig_w, orig_h, x1, y1, box_w, box_h, &padded_w, &padded_h, 0.5f);
                              
                              // Crop saved to best_shots
                              
                              // Minimum face size check
                              if (face_crop.empty() || crop_w < 40 || crop_h < 40) continue;
                              
                              // Aspect ratio check
                              float ratio = (float)crop_h / crop_w;
                              // DEBUG: Log aspect ratios for false positive analysis
                              static int ratio_debug_count = 0;
                              if (ratio_debug_count++ < 100) {
                                  std::cerr << "[C++-RATIO] " << cam_id << " Track=" << track_id 
                                            << " ratio=" << ratio << " conf=" << confidence 
                                            << " box=" << box_w << "x" << box_h << std::endl;
                              }
                              // [FIX] Tighter aspect ratio filter: 1.0 - 1.8 (face-only)
                              if (ratio < 1.0f || ratio > 1.8f) continue;
                              
                              // [NEW] Update track timing
                              auto now = std::chrono::system_clock::now();
                              contextState.track_last_seen[track_id] = now;
                              if (contextState.track_first_seen.find(track_id) == contextState.track_first_seen.end()) {
                                  contextState.track_first_seen[track_id] = now;
                              }
                              
                              // [NEW] Count ALL detections (both <0.7 and >=0.7) for this track
                              contextState.track_detection_count[track_id]++;
                              
                              // [NEW] Only process best shot if confidence >= 0.80
                              // [FIX] Raised threshold to reduce false positives significantly
                              if (confidence >= 0.80f) {
                                  float sharpness = calculate_sharpness_sobel(face_crop, crop_w, crop_h);
                                  float brightness = calculate_brightness(face_crop);
                                  float sqrt_sharpness = std::sqrt(sharpness);
                                  float quality_score = (confidence * 20.0f) + sqrt_sharpness;

                                  if (contextState.best_shots.find(track_id) == contextState.best_shots.end() || 
                                      quality_score > contextState.best_shots[track_id].quality_score) {
                                      BestShotFrame bs;
                                      bs.camera_id = cam_id; bs.track_id = track_id;
                                      bs.frame_data = face_crop; bs.padded_frame_data = face_crop_padded;
                                      bs.width = crop_w; bs.height = crop_h;
                                      bs.padded_width = padded_w; bs.padded_height = padded_h;
                                      bs.quality_score = quality_score; bs.timestamp = now;
                                      bs.confidence = confidence; bs.sharpness = sqrt_sharpness; bs.brightness = brightness;
                                      contextState.best_shots[track_id] = bs;
                                  }
                              }
                          }
                     }
                }
         }
    }

    void ProcessFrame(const char* camera_id, unsigned char* frame_data, int width, int height) {
        if (!frame_data) return;
        unsigned char* batch[1] = { frame_data };
        char* ids[1] = { (char*)camera_id };
        int widths[1] = { width };
        int heights[1] = { height };
        ProcessBatch(ids, batch, 1, widths, heights);
    }

    void StopCameraStream(const char* camera_id) {
        if (!camera_id) return;
        std::lock_guard<std::mutex> lock(contexts_mtx);
        camera_contexts.erase(std::string(camera_id));
    }

    void ShutdownProcessor() {
        shutdown_flag_ = true;
        if (timeout_thread_.joinable()) timeout_thread_.join();
        
        if (buffers[0]) cudaFree(buffers[0]);
        if (buffers[1]) cudaFree(buffers[1]);
        if (trt_stream) cudaStreamDestroy(trt_stream);
        if (trt_context) delete trt_context;
        if (trt_engine) delete trt_engine;
        if (trt_runtime) delete trt_runtime;
        
        std::cout << "[C++] TensorRT Processor Shutdown." << std::endl;
    }
}
