#ifndef PROCESSOR_H
#define PROCESSOR_H

// Cgo-nun C++ kodunu anlaya bilməsi üçün `extern "C"` bloku vacibdir.
#ifdef __cplusplus
extern "C" {
#endif

// Go tərəfinə nəticəni (best shot) göndərmək üçün callback funksiya tipi.
// C++, bu funksiyanı çağıraraq JSON məlumatını Go-ya ötürəcək.
typedef void (*DetectionCallback)(const char* camera_id, const char* json_data);

/**
 * @brief AI Nüvəsini başladır, TensorRT modelini yükləyir və callback-i qeydiyyatdan keçirir.
 * @param engine_path TensorRT .engine faylına gedən yol.
 * @param callback Go tərəfindəki callback funksiyasına bir göstərici (pointer).
 * @return 0 - uğurlu, < 0 - xəta.
 */
int InitializeProcessor(const char* engine_path, DetectionCallback callback);


/**
 * @brief Allocates a frame buffer in C++ managed memory (pinned if possible).
 * @param size Size of the buffer in bytes.
 * @return Pointer to the allocated buffer, or NULL if failed.
 */
unsigned char* AllocateFrame(int size);

/**
 * @brief Frees a frame buffer.
 * @param buffer Pointer to the buffer to free.
 */
void FreeFrame(unsigned char* buffer);

/**
 * @brief Processes a batch of frames.
 * @param camera_id Camera ID.
 * @param frame_data_array Array of pointers to frame data.
 * @param batch_size Number of frames in the batch.
 * @param width Width of the frames.
 * @param height Height of the frames.
 */
void ProcessBatch(char** camera_ids, unsigned char** frame_data_array, int batch_size, int* widths, int* heights);

/**
 * @brief Deprecated: Use ProcessBatch instead.
 */
void ProcessFrame(const char* camera_id, unsigned char* frame_data, int width, int height);

/**
 * @brief Stops a camera stream and cleans up resources.
 * @param camera_id Camera ID to stop.
 */
void StopCameraStream(const char* camera_id);

/**
 * @brief Proqram dayandıqda bütün resursları azad edir və nüvəni dayandırır.
 */
void ShutdownProcessor();

#ifdef __cplusplus
}
#endif

#endif // PROCESSOR_H
