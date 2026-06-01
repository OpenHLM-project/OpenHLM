#include <sl/Camera.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/allocators/gstdmabuf.h>
#include <cuda_runtime_api.h>
#include "NvUtils.h"
#include <nvbufsurftransform.h>
#include <iostream>
#include <csignal>

#define WIDTH 1280
#define HEIGHT 720

volatile sig_atomic_t stop_flag = 0;
void signal_handler(int) { stop_flag = 1; }

// not working

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    signal(SIGINT, signal_handler);

    // Init ZED
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = 30;
    if (zed.open(init_params) != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "ZED open failed\n";
        return -1;
    }

    // GStreamer pipeline: appsrc ! nvvidconv ! nvv4l2h264enc ! h264parse ! filesink
    std::string pipeline_str =
        "appsrc name=mysource is-live=true do-timestamp=true format=time "
        "caps=video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=30/1 ! "
        "nvvidconv ! nvv4l2h264enc maxperf-enable=1 bitrate=4000000 ! "
        "h264parse ! filesink location=zed_encoded.h264";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Pipeline error: " << error->message << "\n";
        g_clear_error(&error);
        return -1;
    }

    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");

    // Create RGBA buffer (input)
    NvBufSurfaceCreateParams rgba_params{};
    rgba_params.width = WIDTH;
    rgba_params.height = HEIGHT;
    rgba_params.layout = NVBUF_LAYOUT_PITCH;
    rgba_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    rgba_params.memType = NVBUF_MEM_DEFAULT;
    NvBufSurface* rgba_surface = nullptr;
    NvBufSurfaceCreate(&rgba_surface, 1, &rgba_params);

    // Create NV12 buffer (output)
    NvBufSurfaceCreateParams nv12_params = rgba_params;
    nv12_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    NvBufSurface* nv12_surface = nullptr;
    NvBufSurfaceCreate(&nv12_surface, 1, &nv12_params);

    // Start pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    int frame_id = 0;

    sl::Mat zed_frame;
    while (!stop_flag) {
        if (zed.grab() != sl::ERROR_CODE::SUCCESS) continue;
        zed.retrieveImage(zed_frame, sl::VIEW::LEFT, sl::MEM::GPU);

        void* src_ptr = zed_frame.getPtr<sl::uchar1>(sl::MEM::GPU);
        // Copy RGBA from ZED GPU to NvBufSurface RGBA
        cudaMemcpy2D(
            rgba_surface->surfaceList[0].dataPtr,
            rgba_surface->surfaceList[0].pitch,
            static_cast<void*>(src_ptr),
            zed_frame.getStepBytes(sl::MEM::GPU),
            WIDTH * 4,
            HEIGHT,
            cudaMemcpyDeviceToDevice
        );

        // Convert RGBA → NV12
        NvBufSurfTransformParams transf_params{};
        transf_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
        transf_params.transform_filter = NvBufSurfTransformInter_Default;
        NvBufSurfTransform(rgba_surface, nv12_surface, &transf_params);

        // Wrap NV12 buffer as GstBuffer via dmabuf fd
        int dma_fd = nv12_surface->surfaceList[0].bufferDesc;
        GstBuffer* gst_buffer = gst_buffer_new();
        GstMemory* memory = gst_dmabuf_allocator_alloc(nullptr, dma_fd, nv12_surface->surfaceList[0].dataSize);
        gst_buffer_append_memory(gst_buffer, memory);
        GST_BUFFER_PTS(gst_buffer) = gst_util_uint64_scale(frame_id, GST_SECOND, 30);
        GST_BUFFER_DURATION(gst_buffer) = gst_util_uint64_scale(1, GST_SECOND, 30);

        gst_app_src_push_buffer(GST_APP_SRC(appsrc), gst_buffer);
        frame_id++;
    }

    std::cout << "Stopping...\n";
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(pipeline);

    NvBufSurfaceDestroy(rgba_surface);
    NvBufSurfaceDestroy(nv12_surface);
    zed.close();

    return 0;
}
