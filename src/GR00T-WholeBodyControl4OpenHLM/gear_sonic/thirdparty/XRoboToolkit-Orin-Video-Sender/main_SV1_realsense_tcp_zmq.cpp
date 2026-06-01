/**
 * main_SV1_realsense_tcp_zmq.cpp - Video compositing and streaming
 *
 * This program receives SV1 images via ZMQ (port 5555) and RealSense
 * wrist camera images via ZMQ (port 5556), composites them together,
 * and sends the result via TCP/GStreamer to VR for display.
 *
 * Usage:
 *   ./OrinVideoSender_SV1_realsense --listen IP:PORT [--zmq_sv1 ENDPOINT] [--zmq_realsense ENDPOINT]
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <fstream>
#include <glib-unix.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openssl/md5.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>

#include "network_helper.hpp"

// ============================================================================
// SV1 Camera Configuration
// ============================================================================
static const int SV1_RAW_WIDTH = 928;      // Full stereo width
static const int SV1_RAW_HEIGHT = 400;     // Frame height
static const int SV1_FPS = 30;             // Target FPS
static const int SV1_SINGLE_WIDTH = 464;   // Single camera width (928/2)

// ============================================================================
// Network Protocol Structures
// ============================================================================
struct CameraRequestData {
  int width;
  int height;
  int fps;
  int bitrate;
  int enableMvHevc;
  int renderMode;
  int port;
  std::string camera;
  std::string ip;

  CameraRequestData()
      : width(0), height(0), fps(0), bitrate(0), enableMvHevc(0), renderMode(0),
        port(0) {}
};

struct NetworkDataProtocol {
  std::string command;
  int length;
  std::vector<uint8_t> data;

  NetworkDataProtocol() : length(0) {}
  NetworkDataProtocol(const std::string &cmd, const std::vector<uint8_t> &d)
      : command(cmd), length(d.size()), data(d) {}
};

// ============================================================================
// Deserialization Classes
// ============================================================================
class CameraRequestDeserializer {
public:
  static CameraRequestData deserialize(const std::vector<uint8_t> &data) {
    if (data.size() < 10) {
      throw std::invalid_argument("Data is too small for valid camera request");
    }
    size_t offset = 0;
    if (data[offset] != 0xCA || data[offset + 1] != 0xFE) {
      throw std::invalid_argument("Invalid magic bytes");
    }
    offset += 2;
    uint8_t version = data[offset++];
    if (version != 1) {
      throw std::invalid_argument("Unsupported protocol version");
    }
    CameraRequestData result;
    if (offset + 28 > data.size()) {
      throw std::invalid_argument("Data too small for integer fields");
    }
    result.width = readInt32(data, offset);
    result.height = readInt32(data, offset + 4);
    result.fps = readInt32(data, offset + 8);
    result.bitrate = readInt32(data, offset + 12);
    result.enableMvHevc = readInt32(data, offset + 16);
    result.renderMode = readInt32(data, offset + 20);
    result.port = readInt32(data, offset + 24);
    offset += 28;
    result.camera = readCompactString(data, offset);
    result.ip = readCompactString(data, offset);
    return result;
  }
private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + 4 > data.size()) {
      throw std::out_of_range("Not enough data to read int32");
    }
    return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                                (data[offset + 2] << 16) | (data[offset + 3] << 24));
  }
  static std::string readCompactString(const std::vector<uint8_t> &data, size_t &offset) {
    if (offset >= data.size()) {
      throw std::out_of_range("Not enough data to read string length");
    }
    uint8_t length = data[offset++];
    if (length == 0) return std::string();
    if (offset + length > data.size()) {
      throw std::out_of_range("Not enough data to read string content");
    }
    std::string result(reinterpret_cast<const char *>(&data[offset]), length);
    offset += length;
    return result;
  }
};

class NetworkDataProtocolDeserializer {
public:
  static NetworkDataProtocol deserialize(const std::vector<uint8_t> &buffer) {
    if (buffer.size() < 8) {
      throw std::invalid_argument("Buffer too small for valid protocol data");
    }
    size_t offset = 0;
    int32_t commandLength = readInt32(buffer, offset);
    offset += 4;
    if (commandLength < 0 || offset + commandLength > buffer.size()) {
      throw std::invalid_argument("Invalid command length");
    }
    std::string command;
    if (commandLength > 0) {
      command = std::string(reinterpret_cast<const char *>(&buffer[offset]), commandLength);
      size_t nullPos = command.find('\0');
      if (nullPos != std::string::npos) command = command.substr(0, nullPos);
    }
    offset += commandLength;
    if (offset + 4 > buffer.size()) {
      throw std::invalid_argument("Buffer too small for data length");
    }
    int32_t dataLength = readInt32(buffer, offset);
    offset += 4;
    if (dataLength < 0 || offset + dataLength > buffer.size()) {
      throw std::invalid_argument("Invalid data length");
    }
    std::vector<uint8_t> data;
    if (dataLength > 0) {
      data.assign(buffer.begin() + offset, buffer.begin() + offset + dataLength);
    }
    return NetworkDataProtocol(command, data);
  }
private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + 4 > data.size()) throw std::out_of_range("Not enough data");
    return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                                (data[offset + 2] << 16) | (data[offset + 3] << 24));
  }
};


// ============================================================================
// Global Variables
// ============================================================================
CameraRequestData current_camera_config;

std::atomic<bool> stop_requested{false};
std::atomic<bool> streaming_active{false};
std::atomic<bool> encoding_enabled{false};
std::atomic<bool> send_enabled{false};
std::atomic<bool> preview_enabled{false};

std::unique_ptr<std::thread> listen_thread;
std::unique_ptr<std::thread> streaming_thread;
std::mutex config_mutex;
std::condition_variable streaming_cv;
std::mutex streaming_mutex;

std::unique_ptr<TCPClient> sender_ptr;
std::unique_ptr<TCPServer> server_ptr;
std::string send_to_server = "";
int send_to_port = 0;

void* zmq_context = nullptr;
void* zmq_sv1_subscriber = nullptr;
void* zmq_realsense_subscriber = nullptr;
std::string zmq_sv1_endpoint = "tcp://192.168.123.164:5555";
std::string zmq_realsense_endpoint = "tcp://192.168.123.164:5556";
std::mutex zmq_mutex;


// ============================================================================
// Helper Functions
// ============================================================================
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr && !stop_requested.load()) {
    try {
      sender_ptr = std::unique_ptr<TCPClient>(new TCPClient(send_to_server, send_to_port));
      std::cout << "Connecting to " << send_to_server << ":" << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

static void* create_sub_socket(void* ctx, const std::string &endpoint, const std::string &name) {
  void* sock = zmq_socket(ctx, ZMQ_SUB);
  if (!sock) {
    std::cerr << "Failed to create ZMQ SUB socket for " << name << std::endl;
    return nullptr;
  }
  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);
  int rcv_hwm = 1;
  zmq_setsockopt(sock, ZMQ_RCVHWM, &rcv_hwm, sizeof(rcv_hwm));
  int conflate = 1;
  zmq_setsockopt(sock, ZMQ_CONFLATE, &conflate, sizeof(conflate));
  int timeout = 1; // 1ms
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  if (zmq_connect(sock, endpoint.c_str()) != 0) {
    std::cerr << "Failed to connect " << name << " SUB to " << endpoint << std::endl;
    zmq_close(sock);
    return nullptr;
  }
  std::cout << "ZMQ " << name << " subscriber connected to " << endpoint << std::endl;
  return sock;
}

bool initialize_zmq() {
  try {
    std::lock_guard<std::mutex> lock(zmq_mutex);
    zmq_context = zmq_ctx_new();
    if (!zmq_context) {
      std::cerr << "Failed to create ZMQ context" << std::endl;
      return false;
    }
    zmq_sv1_subscriber = create_sub_socket(zmq_context, zmq_sv1_endpoint, "SV1");
    if (!zmq_sv1_subscriber) return false;
    zmq_realsense_subscriber = create_sub_socket(zmq_context, zmq_realsense_endpoint, "RealSense");
    if (!zmq_realsense_subscriber) return false;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "ZMQ init error: " << e.what() << std::endl;
    return false;
  }
}

void cleanup_zmq() {
  std::lock_guard<std::mutex> lock(zmq_mutex);
  if (zmq_sv1_subscriber) { zmq_close(zmq_sv1_subscriber); zmq_sv1_subscriber = nullptr; }
  if (zmq_realsense_subscriber) { zmq_close(zmq_realsense_subscriber); zmq_realsense_subscriber = nullptr; }
  if (zmq_context) { zmq_ctx_destroy(zmq_context); zmq_context = nullptr; }
  std::cout << "ZMQ cleaned up" << std::endl;
}

// Forward declarations
void handleOpenCamera(const std::vector<uint8_t> &data);
void handleCloseCamera(const std::vector<uint8_t> &data);
void startStreamingThread();
void stopStreamingThread();
void streamingThreadFunction();
void listenThreadFunction(const std::string &listen_address);

void onDataCallback(const std::string &command) {
  std::vector<uint8_t> binaryData(command.begin(), command.end());
  if (binaryData.size() < 4) {
    std::cerr << "Data too small" << std::endl;
    return;
  }
  uint32_t bodyLength = (static_cast<uint32_t>(binaryData[0]) << 24) |
                        (static_cast<uint32_t>(binaryData[1]) << 16) |
                        (static_cast<uint32_t>(binaryData[2]) << 8) |
                        static_cast<uint32_t>(binaryData[3]);
  if (4 + bodyLength > binaryData.size()) {
    std::cerr << "Invalid body length" << std::endl;
    return;
  }
  std::vector<uint8_t> protocolData(binaryData.begin() + 4, binaryData.begin() + 4 + bodyLength);
  try {
    NetworkDataProtocol protocol = NetworkDataProtocolDeserializer::deserialize(protocolData);
    std::cout << "Command: " << protocol.command << std::endl;
    if (protocol.command == "OPEN_CAMERA") {
      handleOpenCamera(protocol.data);
    } else if (protocol.command == "CLOSE_CAMERA") {
      handleCloseCamera(protocol.data);
    }
  } catch (const std::exception &e) {
    std::cout << "Parse error: " << e.what() << std::endl;
  }
}

void onDisconnectCallback() {
  std::cout << "Client disconnected" << std::endl;
  stopStreamingThread();
}

void listenThreadFunction(const std::string &listen_address) {
  std::cout << "Listen thread started on " << listen_address << std::endl;
  while (!stop_requested.load()) {
    try {
      server_ptr = make_unique_helper<TCPServer>(listen_address);
      server_ptr->setDataCallback(onDataCallback);
      server_ptr->setDisconnectCallback(onDisconnectCallback);
      server_ptr->start();
      std::cout << "TCPServer listening on " << listen_address << std::endl;
      while (!stop_requested.load() && server_ptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (server_ptr) { server_ptr->stop(); server_ptr = nullptr; }
      if (!stop_requested.load()) {
        std::cout << "Waiting for new connection..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    } catch (const std::exception &e) {
      std::cerr << "Listen error: " << e.what() << std::endl;
      if (!stop_requested.load()) std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
  std::cout << "Listen thread stopped" << std::endl;
}

void handle_sigint(int) {
  std::cout << "\nSIGINT received. Stopping..." << std::endl;
  stop_requested.store(true);
  stopStreamingThread();
  if (server_ptr) { server_ptr->stop(); server_ptr = nullptr; }
  cleanup_zmq();
  streaming_cv.notify_all();
}

// GStreamer callback for encoded frames (sends to VR via TCP)
GstFlowReturn on_new_sample(GstAppSink *sink, gpointer /*user_data*/) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_ERROR;

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    gsize size = map.size;

    // TCP send to VR
    if (send_enabled.load() && sender_ptr && sender_ptr->isConnected() && data && size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size) & 0xFF;
        std::copy(data, data + size, packet.begin() + 4);
        sender_ptr->sendData(packet);
      } catch (const TCPException &e) {
        std::cerr << "TCP error: " << e.what() << std::endl;
        streaming_active.store(false);
      }
    }
    gst_buffer_unmap(buffer, &map);
  }
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

void handleOpenCamera(const std::vector<uint8_t> &data) {
  std::cout << "Handling OPEN_CAMERA command" << std::endl;
  try {
    CameraRequestData config = CameraRequestDeserializer::deserialize(data);
    std::cout << "Config - Width: " << config.width << ", Height: " << config.height
              << ", FPS: " << config.fps << ", Bitrate: " << config.bitrate
              << ", EnableMvHevc: " << config.enableMvHevc << ", RenderMode: " << config.renderMode
              << ", IP: " << config.ip << ", Port: " << config.port << std::endl;

    // Accept both "SV1" and "ZED" camera types for compatibility
    if (config.camera != "SV1" && config.camera != "ZED") {
      std::cout << "Camera type: " << config.camera << " (treating as SV1)" << std::endl;
    }

    {
      std::lock_guard<std::mutex> lock(config_mutex);
      current_camera_config = config;
    }
    send_to_server = config.ip;
    send_to_port = config.port;
    std::cout << "Sender target: " << send_to_server << ":" << send_to_port << std::endl;
    startStreamingThread();
  } catch (const std::exception &e) {
    std::cerr << "Failed to parse config: " << e.what() << std::endl;
    if (!send_to_server.empty() && send_to_port > 0) {
      startStreamingThread();
    }
  }
}

void handleCloseCamera(const std::vector<uint8_t> & /*data*/) {
  std::cout << "Handling CLOSE_CAMERA command" << std::endl;
  stopStreamingThread();
}

void startStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  if (streaming_thread && streaming_thread->joinable()) {
    std::cout << "Streaming thread already running" << std::endl;
    return;
  }
  streaming_active.store(true);
  streaming_thread = make_unique_helper<std::thread>(streamingThreadFunction);
  std::cout << "Started streaming thread" << std::endl;
}

void stopStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  streaming_active.store(false);
  encoding_enabled.store(false);
  send_enabled.store(false);
  if (sender_ptr && sender_ptr->isConnected()) {
    sender_ptr->disconnect();
  }
  sender_ptr = nullptr;
  if (streaming_thread && streaming_thread->joinable()) {
    streaming_cv.notify_all();
    streaming_thread->join();
    streaming_thread = nullptr;
    std::cout << "Stopped streaming thread" << std::endl;
  }
}

std::string buildPipelineString(const CameraRequestData &config, bool preview) {
  int width = config.width > 0 ? config.width : SV1_RAW_WIDTH;
  int height = config.height > 0 ? config.height : SV1_RAW_HEIGHT;
  int fps = config.fps > 0 ? config.fps : SV1_FPS;
  int bitrate = config.bitrate > 0 ? config.bitrate : 4000000;

  // Detect platform: check if we're on Jetson (ARM) or x86_64
  bool is_jetson = false;
  #ifdef __aarch64__
    is_jetson = true;
  #endif

  std::string encoder, parser, converter, pipeline;

  if (is_jetson) {
    // Jetson platform: use hardware accelerated encoding
    encoder = config.enableMvHevc ? "nvv4l2h265enc" : "nvv4l2h264enc";
    parser = config.enableMvHevc ? "h265parse" : "h264parse";
    converter = "nvvidconv ! video/x-raw(memory:NVMM),format=NV12";

    pipeline = "appsrc name=mysource is-live=true format=time "
        "caps=video/x-raw,format=BGRA,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! " + converter + " ! tee name=t "
        "t. ! queue ! " + encoder + " maxperf-enable=1 insert-sps-pps=true "
        "idrinterval=15 bitrate=" + std::to_string(bitrate) + " ! " +
        parser + " ! appsink name=mysink emit-signals=true sync=false ";

    if (preview) {
      pipeline += "t. ! queue ! nvvidconv ! videoconvert ! autovideosink sync=false ";
    }
  } else {
    // x86_64 platform: use software encoding (x264enc)
    // Match Jetson configuration as closely as possible
    encoder = config.enableMvHevc ? "x265enc" : "x264enc";
    parser = config.enableMvHevc ? "h265parse" : "h264parse";

    // Match Jetson's IDR interval (15 frames) and bitrate settings
    int idr_interval = 15;

    // Build main pipeline with tee (for potential preview branch)
    pipeline = "appsrc name=mysource is-live=true format=time "
        "caps=video/x-raw,format=BGRA,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! video/x-raw,format=NV12 ! tee name=t "
        "t. ! queue ! " + encoder + " speed-preset=ultrafast tune=zerolatency "
        "key-int-max=" + std::to_string(idr_interval) + " bitrate=" + std::to_string(bitrate / 1000) +
        " bframes=0 ! " +
        parser + " ! video/x-h264,stream-format=byte-stream,alignment=au ! "
        "appsink name=mysink emit-signals=true sync=false ";

    if (preview) {
      pipeline += "t. ! queue ! videoconvert ! autovideosink sync=false ";
    }
  }

  return pipeline;
}

void streamingThreadFunction() {
  std::cout << "Streaming thread started" << std::endl;

  try {
    // Initialize TCP sender
    bool tcp_initialized = false;
    if (!send_to_server.empty() && send_to_port > 0) {
      tcp_initialized = initialize_sender();
      if (!tcp_initialized) {
        std::cerr << "TCP init failed, continuing with ZMQ only" << std::endl;
      }
    }

    encoding_enabled.store(true);
    if (tcp_initialized) send_enabled.store(true);

    if (!send_enabled.load()) {
      std::cerr << "No output method available" << std::endl;
      return;
    }

    // Get config
    CameraRequestData config;
    {
      std::lock_guard<std::mutex> lock(config_mutex);
      config = current_camera_config;
    }

    // Build GStreamer pipeline
    std::string pipeline_str = buildPipelineString(config, preview_enabled.load());
    std::cout << "Pipeline: " << pipeline_str << std::endl;

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
      std::cerr << "Pipeline error: " << error->message << std::endl;
      g_clear_error(&error);
      return;
    }

    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    cv::Mat left_img, right_img;
    int frame_id = 0;

    // Persistent wrist camera frames to avoid flickering
    cv::Mat last_left_wrist, last_right_wrist;

    std::cout << "Starting streaming loop..." << std::endl;
    std::cout << "Encoding enabled: " << (encoding_enabled.load() ? "YES" : "NO") << std::endl;
    while (streaming_active.load() && !stop_requested.load()) {
      // Receive SV1 image via ZMQ (format: 12-byte header [width][height][channels] + raw BGR data)
      bool sv1_received = false;
      try {
        zmq_msg_t message;
        zmq_msg_init(&message);
        int rc = zmq_msg_recv(&message, zmq_sv1_subscriber, ZMQ_DONTWAIT);
        if (rc != -1) {
          size_t total_size = zmq_msg_size(&message);
          uint8_t* data_ptr = (uint8_t*)zmq_msg_data(&message);
          if (total_size >= 12) {
            int32_t width, height, channels;
            std::memcpy(&width, &data_ptr[0], 4);
            std::memcpy(&height, &data_ptr[4], 4);
            std::memcpy(&channels, &data_ptr[8], 4);
            size_t expected = 12 + (size_t)width * height * channels;
            if (total_size >= expected && width > 0 && height > 0 && channels > 0) {
              int cv_type = (channels == 3) ? CV_8UC3 : CV_8UC1;
              cv::Mat raw(height, width, cv_type, data_ptr + 12);
              int half_w = width / 2;
              left_img = raw(cv::Rect(0, 0, half_w, height)).clone();
              right_img = raw(cv::Rect(half_w, 0, half_w, height)).clone();
              sv1_received = true;
            }
          }
        }
        zmq_msg_close(&message);
      } catch (const std::exception &e) {
        std::cerr << "ZMQ SV1 receive error: " << e.what() << std::endl;
      }

      if (!sv1_received) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      // Try to receive new wrist camera frames via RealSense ZMQ
      try {
        zmq_msg_t message;
        zmq_msg_init(&message);
        int rc = zmq_msg_recv(&message, zmq_realsense_subscriber, ZMQ_DONTWAIT);

        if (rc != -1) {
            size_t total_size = zmq_msg_size(&message);
            uint8_t* data_ptr = (uint8_t*)zmq_msg_data(&message);

            if (total_size >= 12) {
                // Parse header: [width][height][jpeg_len]
                int32_t width, height, jpeg_len;
                std::memcpy(&width, &data_ptr[0], 4);
                std::memcpy(&height, &data_ptr[4], 4);
                std::memcpy(&jpeg_len, &data_ptr[8], 4);

                // Validate JPEG data size
                if (jpeg_len > 0 && total_size >= 12 + static_cast<size_t>(jpeg_len)) {
                    // Decode JPEG data
                    std::vector<uint8_t> jpeg_data(data_ptr + 12, data_ptr + 12 + jpeg_len);
                    cv::Mat combined_wrist = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);

                    if (!combined_wrist.empty()) {
                        int half_w_wrist = combined_wrist.cols / 2;

                        // Update persistent wrist frames (only when new data arrives)
                        cv::resize(combined_wrist(cv::Rect(0, 0, half_w_wrist, combined_wrist.rows)),
                                  last_left_wrist, cv::Size(360, 270));
                        cv::resize(combined_wrist(cv::Rect(half_w_wrist, 0, half_w_wrist, combined_wrist.rows)),
                                  last_right_wrist, cv::Size(360, 270));
                    }
                }
            }
        }
        zmq_msg_close(&message);
      } catch (const std::exception &e) {
        std::cerr << "ZMQ RealSense receive error: " << e.what() << std::endl;
      }
     
      // Process images for VR (GStreamer/TCP)
      if (encoding_enabled.load()) {
        cv::Mat head_camera, vr_image;

        // Raw mode: concatenate left with left (same as original behavior)
        cv::Mat raw_stereo;
        cv::hconcat(right_img, right_img, raw_stereo);
        cv::flip(raw_stereo, raw_stereo, -1);
        cv::rotate(raw_stereo, raw_stereo, cv::ROTATE_180);
        vr_image = raw_stereo;

        int half_w = vr_image.cols / 2;
        int h = vr_image.rows;
        head_camera = vr_image(cv::Rect(0, 0, half_w, h));
        cv::resize(head_camera, head_camera, cv::Size(1080, 810));

        // Cobine three pictures into a big picture for teleoperator
        cv::Mat big_canvas = head_camera;

        // Copy wrist images if available (use last frame to avoid flickering)
        if (!last_left_wrist.empty() && !last_right_wrist.empty()) {
          last_left_wrist.copyTo(big_canvas(cv::Rect(0, 0, 360, 270)));
          last_right_wrist.copyTo(big_canvas(cv::Rect(720, 0, 360, 270)));
        }

        vr_image = big_canvas;
        cv::hconcat(vr_image, vr_image, vr_image);

        // Convert to BGRA for GStreamer
        cv::Mat bgra;
        cv::cvtColor(vr_image, bgra, cv::COLOR_BGR2BGRA);

        // Push to GStreamer
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, bgra.total() * bgra.elemSize(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, bgra.data, bgra.total() * bgra.elemSize());
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame_id, GST_SECOND, SV1_FPS);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, SV1_FPS);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        frame_id++;
      }
    }

    std::cout << "Streaming loop ended, cleaning up..." << std::endl;
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

  } catch (const std::exception &e) {
    std::cerr << "Streaming error: " << e.what() << std::endl;
  }
  std::cout << "Streaming thread finished" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  bool preview_local = false;
  bool listen_enabled = false;
  std::string listen_address = "";

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_local = true;
    } else if (arg == "--listen" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
    } else if (arg == "--zmq_sv1" && i + 1 < argc) {
      zmq_sv1_endpoint = argv[++i];
    } else if (arg == "--zmq_realsense" && i + 1 < argc) {
      zmq_realsense_endpoint = argv[++i];
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --preview              Enable video preview\n"
                << "  --listen ADDR          Listen on address (IP:PORT)\n"
                << "  --zmq_sv1 ENDPOINT     ZMQ endpoint for SV1 images (default: tcp://192.168.123.164:5555)\n"
                << "  --zmq_realsense ENDPOINT  ZMQ endpoint for RealSense images (default: tcp://192.168.123.164:5556)\n";
      return 0;
    }
  }

  if (!listen_enabled) {
    std::cerr << "Error: --listen required\n";
    return -1;
  }

  // Initialize ZMQ subscribers
  if (!initialize_zmq()) {
    std::cerr << "ZMQ init failed\n";
    return -1;
  }
  std::cout << "ZMQ initialized: SV1=" << zmq_sv1_endpoint
            << " RealSense=" << zmq_realsense_endpoint << std::endl;

  preview_enabled.store(preview_local);

  std::cout << "Starting listen mode on " << listen_address << "\n";
  listen_thread = make_unique_helper<std::thread>(listenThreadFunction, listen_address);
  std::cout << "Press Ctrl+C to stop.\n";
  while (!stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (listen_thread && listen_thread->joinable()) {
    listen_thread->join();
  }

  std::cout << "Shutting down...\n";
  stopStreamingThread();
  cleanup_zmq();
  std::cout << "Done.\n";
  return 0;
}
