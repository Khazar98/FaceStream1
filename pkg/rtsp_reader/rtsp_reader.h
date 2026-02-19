#ifndef RTSP_READER_H
#define RTSP_READER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

typedef struct {
    AVFormatContext *format_ctx;
    AVCodecContext *codec_ctx;
    AVFrame *frame;
    AVFrame *frame_bgr;
    AVPacket *packet;
    struct SwsContext *sws_ctx;
    int video_stream_idx;
    int width;
    int height;
} RtspContext;

/**
 * @brief Opens an RTSP stream.
 * @param url RTSP URL.
 * @return Pointer to RtspContext or NULL on failure.
 */
RtspContext* open_rtsp_stream(const char* url);

/**
 * @brief Reads a frame from the stream and writes it to the provided buffer.
 * @param ctx Pointer to RtspContext.
 * @param buffer Pointer to the output buffer (must be pre-allocated).
 * @param buffer_size Size of the buffer.
 * @return 0 on success, < 0 on failure/EOF.
 */
int read_frame(RtspContext* ctx, unsigned char* buffer, int buffer_size);

/**
 * @brief Closes the RTSP stream and frees resources.
 * @param ctx Pointer to RtspContext.
 */
void close_rtsp_stream(RtspContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // RTSP_READER_H
