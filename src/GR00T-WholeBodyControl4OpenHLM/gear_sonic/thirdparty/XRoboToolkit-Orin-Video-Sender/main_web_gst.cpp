#include <algorithm>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fcntl.h>
#include <glib-unix.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <iostream>
#include <memory>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "network_helper.hpp"

std::unique_ptr<TCPClient> sender_ptr;
GMainLoop *loop = nullptr;
volatile sig_atomic_t stop_requested = 0;
bool send_enabled = false; // <-- Global flag for sending

static void signal_handler(int sig) {
  if (!stop_requested && loop) {
    stop_requested = 1;
    g_main_loop_quit(loop);
  }
}

void printErrorAndQuit(const std::string &error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
  if (loop) {
    g_main_loop_quit(loop);
  }
}

GstFlowReturn on_new_sample(GstAppSink *sink, gpointer user_data) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample)
    return GST_FLOW_ERROR;

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    gsize size = map.size;
    if (send_enabled && sender_ptr && sender_ptr->isConnected() && data &&
        size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size)&0xFF;
        std::copy(data, data + size, packet.begin() + 4);
        sender_ptr->sendData(packet);
        std::cout << "Sent " << size << " bytes of H.264 data" << std::endl;
      } catch (const TCPException &e) {
        printErrorAndQuit(e.what());
      } catch (const std::exception &e) {
        printErrorAndQuit("Unexpected error during sendData: " +
                          std::string(e.what()));
      }
    }

    gst_buffer_unmap(buffer, &map);
  }

  if (buffer) {
    GstClockTime timestamp = GST_BUFFER_PTS(buffer);
    std::cout << "Encoded frame at timestamp: "
              << GST_TIME_AS_MSECONDS(timestamp) << " ms" << std::endl;
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  g_unix_signal_add(
      SIGINT,
      [](gpointer user_data) -> gboolean {
        if (loop && !stop_requested) {
          stop_requested = 1;
          g_main_loop_quit(loop);
        }
        return G_SOURCE_REMOVE;
      },
      nullptr);

  bool preview_enabled = false;
  std::string server_ip = "127.0.0.1";
  int server_port = 12345;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_enabled = true;
    } else if (arg == "--send") {
      send_enabled = true;
    } else if (arg == "--server" && i + 1 < argc) {
      server_ip = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      server_port = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --preview      Enable video preview\n";
      std::cout << "  --send         Enable sending encoded video over TCP\n";
      std::cout << "  --server IP    Server IP address (default: 127.0.0.1)\n";
      std::cout << "  --port PORT    Server port (default: 12345)\n";
      std::cout << "  --help         Show this help message\n";
      return 0;
    }
  }

  if (send_enabled) {
    try {
      sender_ptr =
          std::unique_ptr<TCPClient>(new TCPClient(server_ip, server_port));
      std::cout << "Attempting to connect to " << server_ip << ":"
                << server_port << std::endl;
      sender_ptr->connect();
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect to server: " << e.what() << std::endl;
      return -1;
    }
  }

  std::string pipeline_desc;

  if (preview_enabled) {
    pipeline_desc =
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=1280,height=720 ! "
        "videoconvert ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        "tee name=t "
        "t. ! queue ! nvv4l2h264enc maxperf-enable=1 bitrate=4000000 ! "
        "h264parse ! appsink name=mysink emit-signals=true sync=false "
        "t. ! queue ! nvvidconv ! video/x-raw ! videoconvert ! autovideosink "
        "sync=false";
  } else {
    pipeline_desc =
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=1280,height=720 ! "
        "videoconvert ! "
        "nvvidconv ! "
        "nvv4l2h264enc maxperf-enable=1 bitrate=4000000 ! "
        "h264parse ! appsink name=mysink emit-signals=true sync=false";
  }

  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
  if (!pipeline) {
    g_printerr("Failed to create pipeline: %s\n", error->message);
    g_clear_error(&error);
    return -1;
  }

  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  if (!sink) {
    g_printerr("Failed to get appsink element\n");
    gst_object_unref(pipeline);
    return -1;
  }

  g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), nullptr);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_print("Capturing and encoding... Press Ctrl+C to stop.\n");

  loop = g_main_loop_new(nullptr, FALSE);
  g_main_loop_run(loop);
  g_print("\nStopping pipeline...\n");

  if (send_enabled && sender_ptr) {
    sender_ptr->disconnect();
    std::cout << "Disconnected from server" << std::endl;
  }

  gst_element_send_event(pipeline, gst_event_new_eos());
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  gst_object_unref(sink);
  g_main_loop_unref(loop);
  loop = nullptr;

  return 0;
}
