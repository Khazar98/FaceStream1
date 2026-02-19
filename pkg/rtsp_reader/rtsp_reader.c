#include "rtsp_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

RtspContext* open_rtsp_stream(const char* url) {
    avformat_network_init();
    
    RtspContext* ctx = (RtspContext*)calloc(1, sizeof(RtspContext));
    if (!ctx) return NULL;

    AVDictionary* options = NULL;
    av_dict_set(&options, "rtsp_transport", "tcp", 0); // Force TCP for reliability
    av_dict_set(&options, "buffer_size", "1024000", 0);
    av_dict_set(&options, "max_delay", "500000", 0); // 0.5s max delay

    if (avformat_open_input(&ctx->format_ctx, url, NULL, &options) != 0) {
        fprintf(stderr, "[FFmpeg] Failed to open stream: %s\n", url);
        free(ctx);
        return NULL;
    }

    if (avformat_find_stream_info(ctx->format_ctx, NULL) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to find stream info\n");
        avformat_close_input(&ctx->format_ctx);
        free(ctx);
        return NULL;
    }

    ctx->video_stream_idx = -1;
    for (unsigned int i = 0; i < ctx->format_ctx->nb_streams; i++) {
        if (ctx->format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (ctx->video_stream_idx == -1) {
                ctx->video_stream_idx = i;
                // Found the video stream
                break; 
            }
        }
    }

    if (ctx->video_stream_idx == -1) {
        fprintf(stderr, "[FFmpeg] No video stream found\n");
        avformat_close_input(&ctx->format_ctx);
        free(ctx);
        return NULL;
    }

    AVCodecParameters* codecpar = ctx->format_ctx->streams[ctx->video_stream_idx]->codecpar;
    
    // [CUDA] Try to find hardware decoder first
    const AVCodec* codec = NULL;
    if (codecpar->codec_id == AV_CODEC_ID_H264) {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (codec) fprintf(stderr, "[FFmpeg] Found h264_cuvid decoder\n");
    } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
        codec = avcodec_find_decoder_by_name("hevc_cuvid");
        if (codec) fprintf(stderr, "[FFmpeg] Found hevc_cuvid decoder\n");
    }

    // Fallback to default decoder
    if (!codec) {
        fprintf(stderr, "[FFmpeg] CUDA decoder not found, falling back to software\n");
        codec = avcodec_find_decoder(codecpar->codec_id);
    }
    if (!codec) {
        fprintf(stderr, "[FFmpeg] Codec not found\n");
        avformat_close_input(&ctx->format_ctx);
        free(ctx);
        return NULL;
    }

    ctx->codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(ctx->codec_ctx, codecpar);

    if (avcodec_open2(ctx->codec_ctx, codec, NULL) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to open codec\n");
        avcodec_free_context(&ctx->codec_ctx);
        avformat_close_input(&ctx->format_ctx);
        free(ctx);
        return NULL;
    }

    ctx->frame = av_frame_alloc();
    ctx->frame_bgr = av_frame_alloc();
    ctx->packet = av_packet_alloc();
    
    ctx->width = ctx->codec_ctx->width;
    ctx->height = ctx->codec_ctx->height;

    // Prepare scaler for YUV -> BGR conversion
    // Use native resolution (no resize here). Resizing will be done in AI Processor (Letterbox).
    int target_w = ctx->codec_ctx->width;
    int target_h = ctx->codec_ctx->height;
    fprintf(stderr, "[RTSP] Camera Resolution: %dx%d (Native)\n", target_w, target_h);

    // Allocate buffer for BGR frame (managed by caller/FFmpeg context)
    // We just set up the scaler here.

    ctx->sws_ctx = sws_getContext(
        ctx->codec_ctx->width, ctx->codec_ctx->height, ctx->codec_ctx->pix_fmt,
        target_w, target_h, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    ctx->width = target_w;
    ctx->height = target_h;

    return ctx;
}

int read_frame(RtspContext* ctx, unsigned char* buffer, int buffer_size) {
    if (!ctx) return -1;

    while (av_read_frame(ctx->format_ctx, ctx->packet) >= 0) {
        if (ctx->packet->stream_index == ctx->video_stream_idx) {
            int ret = avcodec_send_packet(ctx->codec_ctx, ctx->packet);
            if (ret < 0) {
                av_packet_unref(ctx->packet);
                continue;
            }

            ret = avcodec_receive_frame(ctx->codec_ctx, ctx->frame);
            if (ret == 0) {
                // [DEBUG] Check for Dynamic Resolution Change
                if (ctx->frame->width != ctx->width || ctx->frame->height != ctx->height) {
                    fprintf(stderr, "[RTSP] Resolution Changed! Init: %dx%d, Frame: %dx%d. This will cause corruption!\n", 
                            ctx->width, ctx->height, ctx->frame->width, ctx->frame->height);
                }

                // Determine target buffer size
                int required_size = ctx->width * ctx->height * 3;
                if (buffer_size < required_size) {
                    fprintf(stderr, "[FFmpeg] Buffer too small: %d < %d\n", buffer_size, required_size);
                    av_packet_unref(ctx->packet);
                    return -2;
                }

                // Map the output buffer to the AVFrame structure
                av_image_fill_arrays(
                    ctx->frame_bgr->data, ctx->frame_bgr->linesize,
                    buffer, AV_PIX_FMT_BGR24, ctx->width, ctx->height, 1
                );
                
                // [DEBUG] Check stride
                if (ctx->frame_bgr->linesize[0] != ctx->width * 3) {
                     fprintf(stderr, "[RTSP] Stride Mismatch! W: %d, Expected: %d, Got: %d\n", ctx->width, ctx->width*3, ctx->frame_bgr->linesize[0]);
                }

                // Convert/Scale to BGR
                sws_scale(
                    ctx->sws_ctx,
                    (const uint8_t * const *)ctx->frame->data, ctx->frame->linesize,
                    0, ctx->codec_ctx->height,
                    ctx->frame_bgr->data, ctx->frame_bgr->linesize
                );

                av_packet_unref(ctx->packet);
                return 0; // Success
            }
        }
        av_packet_unref(ctx->packet);
    }
    return -1; // EOF or error
}

void close_rtsp_stream(RtspContext* ctx) {
    if (!ctx) return;
    
    if (ctx->sws_ctx) sws_freeContext(ctx->sws_ctx);
    if (ctx->frame) av_frame_free(&ctx->frame);
    if (ctx->frame_bgr) av_frame_free(&ctx->frame_bgr);
    if (ctx->packet) av_packet_free(&ctx->packet);
    if (ctx->codec_ctx) avcodec_free_context(&ctx->codec_ctx);
    if (ctx->format_ctx) avformat_close_input(&ctx->format_ctx);
    
    free(ctx);
}
