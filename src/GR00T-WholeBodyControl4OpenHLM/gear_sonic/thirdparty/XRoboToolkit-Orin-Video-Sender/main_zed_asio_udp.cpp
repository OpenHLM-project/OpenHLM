#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <glib-unix.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openssl/md5.h>
#include <sl/Camera.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "network_asio.hpp"

// Network Protocol Structures
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
      : command(cmd), data(d), length(d.size()) {}
};

// Deserialization functions
class CameraRequestDeserializer {
public:
  static CameraRequestData deserialize(const std::vector<uint8_t> &data) {
    if (data.size() < 10) {
      throw std::invalid_argument("Data is too small for valid camera request");
    }

    size_t offset = 0;

    // Check magic bytes (0xCA, 0xFE)
    if (data[offset] != 0xCA || data[offset + 1] != 0xFE) {
      throw std::invalid_argument("Invalid magic bytes");
    }
    offset += 2;

    // Check protocol version
    uint8_t version = data[offset++];
    if (version != 1) {
      throw std::invalid_argument("Unsupported protocol version");
    }

    CameraRequestData result;

    // Read integer fields (7 * 4 bytes)
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

    // Read strings with compact encoding
    result.camera = readCompactString(data, offset);
    result.ip = readCompactString(data, offset);

    return result;
  }

private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + 4 > data.size()) {
      throw std::out_of_range("Not enough data to read int32");
    }

    // Little-endian format (matching C# BitConverter default)
    return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                                (data[offset + 2] << 16) |
                                (data[offset + 3] << 24));
  }

  static std::string readCompactString(const std::vector<uint8_t> &data,
                                       size_t &offset) {
    if (offset >= data.size()) {
      throw std::out_of_range("Not enough data to read string length");
    }

    uint8_t length = data[offset++];
    if (length == 0) {
      return std::string();
    }

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
    if (buffer.size() <
        8) { // Minimum: 4 bytes command length + 4 bytes data length
      throw std::invalid_argument("Buffer too small for valid protocol data");
    }

    size_t offset = 0;

    // Read command length
    int32_t commandLength = readInt32(buffer, offset);
    offset += 4;

    if (commandLength < 0 || offset + commandLength > buffer.size()) {
      throw std::invalid_argument("Invalid command length");
    }

    // Read command
    std::string command;
    if (commandLength > 0) {
      command = std::string(reinterpret_cast<const char *>(&buffer[offset]),
                            commandLength);
      // Remove any null terminators
      size_t nullPos = command.find('\0');
      if (nullPos != std::string::npos) {
        command = command.substr(0, nullPos);
      }
    }
    offset += commandLength;

    if (offset + 4 > buffer.size()) {
      throw std::invalid_argument("Buffer too small for data length");
    }

    // Read data length
    int32_t dataLength = readInt32(buffer, offset);
    offset += 4;

    if (dataLength < 0 || offset + dataLength > buffer.size()) {
      throw std::invalid_argument("Invalid data length");
    }

    // Read data
    std::vector<uint8_t> data;
    if (dataLength > 0) {
      data.assign(buffer.begin() + offset,
                  buffer.begin() + offset + dataLength);
    }

    return NetworkDataProtocol(command, data);
  }

private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + 4 > data.size()) {
      throw std::out_of_range("Not enough data to read int32");
    }

    // Little-endian format
    return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                                (data[offset + 2] << 16) |
                                (data[offset + 3] << 24));
  }
};

// Global camera configuration
CameraRequestData current_camera_config;

// Thread-safe global state
std::atomic<bool> stop_requested{false};
std::atomic<bool> streaming_active{false};
std::atomic<bool> encoding_enabled{false};
std::atomic<bool> send_enabled{false};
std::atomic<bool> preview_enabled{false};

// Thread management
std::unique_ptr<std::thread> listen_thread;
std::unique_ptr<std::thread> streaming_thread;
std::mutex config_mutex;
std::condition_variable streaming_cv;
std::mutex streaming_mutex;

// Network components
std::unique_ptr<UDPClient> sender_ptr;
std::unique_ptr<TCPServer> server_ptr;
std::string send_to_server = "";
int send_to_port = 0;

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr && !stop_requested.load()) {
    try {
      sender_ptr = std::unique_ptr<UDPClient>(
          new UDPClient(send_to_server, send_to_port));
      std::cout << "Attempting to connect to " << send_to_server << ":"
                << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect to server: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    // Sleep for 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

// Template helper for C++11 make_unique replacement
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Forward declarations
void handleOpenCamera(const std::vector<uint8_t> &data);
void handleCloseCamera(const std::vector<uint8_t> &data);
void startStreamingThread();
void stopStreamingThread();
void streamingThreadFunction();
void listenThreadFunction(const std::string &listen_address);

void onDataCallback(const std::string &command) {
  // Convert string to binary data for protocol parsing
  std::vector<uint8_t> binaryData(command.begin(), command.end());

  for (size_t i = 0; i < std::min(binaryData.size(), size_t(32)); ++i) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned int>(binaryData[i]) << " ";
  }
  std::cout << std::dec << std::endl;

  // First, extract the actual protocol data from the 4-byte wrapper
  if (binaryData.size() < 4) {
    std::cerr << "Data too small to contain length header" << std::endl;
    return;
  }

  // Read the 4-byte header (big-endian format) to get the actual data length
  uint32_t bodyLength = (static_cast<uint32_t>(binaryData[0]) << 24) |
                        (static_cast<uint32_t>(binaryData[1]) << 16) |
                        (static_cast<uint32_t>(binaryData[2]) << 8) |
                        static_cast<uint32_t>(binaryData[3]);

  if (4 + bodyLength > binaryData.size()) {
    std::cerr << "Data too small for declared body length. Expected: "
              << (4 + bodyLength) << ", got: " << binaryData.size()
              << std::endl;
    return;
  }

  // Extract the actual protocol data (skip the 4-byte header)
  std::vector<uint8_t> protocolData(binaryData.begin() + 4,
                                    binaryData.begin() + 4 + bodyLength);

  for (size_t i = 0; i < std::min(protocolData.size(), size_t(32)); ++i) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned int>(protocolData[i]) << " ";
  }
  std::cout << std::dec << std::endl;

  // Now try to parse as NetworkDataProtocol format (with length prefixes)
  try {
    // Debug: Print first few fields of the protocol data
    if (protocolData.size() >= 8) {
      int32_t cmdLen = (protocolData[0]) | (protocolData[1] << 8) |
                       (protocolData[2] << 16) | (protocolData[3] << 24);
      int32_t dataLenPos = 4 + cmdLen;
      std::cout << "Command length: " << cmdLen << std::endl;
      if (static_cast<size_t>(dataLenPos + 4) <= protocolData.size()) {
        int32_t dataLen = (protocolData[dataLenPos]) |
                          (protocolData[dataLenPos + 1] << 8) |
                          (protocolData[dataLenPos + 2] << 16) |
                          (protocolData[dataLenPos + 3] << 24);
        std::cout << "Data length: " << dataLen << std::endl;
      }
    }

    NetworkDataProtocol protocol =
        NetworkDataProtocolDeserializer::deserialize(protocolData);

    std::cout << "Received protocol command: '" << protocol.command
              << "' (length: " << protocol.command.length() << ")" << std::endl;

    // Handle the protocol commands
    if (protocol.command == "OPEN_CAMERA") {
      handleOpenCamera(protocol.data);
    } else if (protocol.command == "CLOSE_CAMERA") {
      handleCloseCamera(protocol.data);
    } else {
      std::cout << "Unknown protocol command: " << protocol.command
                << std::endl;
    }
    return;
  } catch (const std::exception &e) {
    // If NetworkDataProtocol parsing fails, try simple command format
    std::cout << "Failed to parse as NetworkDataProtocol: " << e.what()
              << std::endl;
  }
}
///

// Handler function implementations
void handleOpenCamera(const std::vector<uint8_t> &data) {
  std::cout << "Handling OPEN_CAMERA command" << std::endl;

  try {
    // Parse the camera configuration data
    CameraRequestData cameraConfig =
        CameraRequestDeserializer::deserialize(data);

    std::cout << "Camera config - Width: " << cameraConfig.width
              << ", Height: " << cameraConfig.height
              << ", FPS: " << cameraConfig.fps
              << ", Bitrate: " << cameraConfig.bitrate
              << ", IP: " << cameraConfig.ip << ", Port: " << cameraConfig.port
              << ", type: " << cameraConfig.camera << std::endl;
    // Do nothing if the type is not "ZED"
    if (cameraConfig.camera != "ZED") {
      std::cout << "Unsupported camera type: " << cameraConfig.camera
                << ". Only 'ZED' is supported." << std::endl;
      return;
    }

    // Store the camera configuration globally
    {
      std::lock_guard<std::mutex> lock(config_mutex);
      current_camera_config = cameraConfig;
    }

    // Set the global sender connection parameters from config
    send_to_server = cameraConfig.ip;
    send_to_port = cameraConfig.port;

    std::cout << "Updated sender target to " << send_to_server << ":"
              << send_to_port << std::endl;

    // Start the streaming thread which will use the updated config
    startStreamingThread();

  } catch (const std::exception &e) {
    std::cerr << "Failed to parse camera config: " << e.what() << std::endl;
    // Start with default configuration only if we have valid defaults
    if (!send_to_server.empty() && send_to_port > 0) {
      startStreamingThread();
    } else {
      std::cerr
          << "No valid server configuration available, cannot start streaming"
          << std::endl;
    }
  }
}

void handleCloseCamera(const std::vector<uint8_t> &data) {
  std::cout << "Handling CLOSE_CAMERA command" << std::endl;
  stopStreamingThread();
}

void onDisconnectCallback() {
  std::cout << "Client disconnected, stopping streaming" << std::endl;
  stopStreamingThread();
}

void handle_sigint(int) {
  std::cout << "\nSIGINT received. Stopping all threads..." << std::endl;
  stop_requested.store(true);

  // Stop streaming first
  stopStreamingThread();

  // Stop server
  if (server_ptr) {
    server_ptr->stop();
    server_ptr = nullptr;
  }

  // Wake up any waiting threads
  streaming_cv.notify_all();
}

void printErrorAndQuit(const std::string &error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
  stop_requested.store(true);
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

  // Disconnect sender if connected
  if (sender_ptr && sender_ptr->isConnected()) {
    sender_ptr->disconnect();
  }
  sender_ptr = nullptr;

  // Wait for streaming thread to finish
  if (streaming_thread && streaming_thread->joinable()) {
    streaming_cv.notify_all();
    streaming_thread->join();
    streaming_thread = nullptr;
    std::cout << "Stopped streaming thread" << std::endl;
  }
}

void listenThreadFunction(const std::string &listen_address) {
  std::cout << "Listen thread started on " << listen_address << std::endl;

  while (!stop_requested.load()) {
    try {
      // Initialize TCPServer
      server_ptr = std::unique_ptr<TCPServer>(new TCPServer(listen_address));
      server_ptr->setDataCallback(onDataCallback);
      server_ptr->setDisconnectCallback(onDisconnectCallback);
      server_ptr->start();
      std::cout << "TCPServer is listening on " << listen_address << std::endl;

      // Keep the server running until stop is requested
      while (!stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      server_ptr->stop();
      server_ptr = nullptr;
      break;
    } catch (const std::exception &e) {
      std::cerr << "Listen thread error: " << e.what() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  std::cout << "Listen thread stopped" << std::endl;
}

inline cv::Mat slMat2cvMat(sl::Mat &input) {
  // Mapping between MAT_TYPE and CV_TYPE
  int cv_type = -1;
  switch (input.getDataType()) {
  case sl::MAT_TYPE::F32_C1:
    cv_type = CV_32FC1;
    break;
  case sl::MAT_TYPE::F32_C2:
    cv_type = CV_32FC2;
    break;
  case sl::MAT_TYPE::F32_C3:
    cv_type = CV_32FC3;
    break;
  case sl::MAT_TYPE::F32_C4:
    cv_type = CV_32FC4;
    break;
  case sl::MAT_TYPE::U8_C1:
    cv_type = CV_8UC1;
    break;
  case sl::MAT_TYPE::U8_C2:
    cv_type = CV_8UC2;
    break;
  case sl::MAT_TYPE::U8_C3:
    cv_type = CV_8UC3;
    break;
  case sl::MAT_TYPE::U8_C4:
    cv_type = CV_8UC4;
    break;
  default:
    break;
  }

  return cv::Mat(input.getHeight(), input.getWidth(), cv_type,
                 input.getPtr<sl::uchar1>(sl::MEM::CPU));
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
    if (send_enabled.load() && sender_ptr && sender_ptr->isConnected() &&
        data && size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size)&0xFF;
        std::copy(data, data + size, packet.begin() + 4);

        sender_ptr->sendData(packet);
      } catch (const TCPException &e) {
        printErrorAndQuit(e.what());
      } catch (const std::exception &e) {
        printErrorAndQuit("Unexpected error during sendData: " +
                          std::string(e.what()));
      }
    }

    gst_buffer_unmap(buffer, &map);
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

void streamingThreadFunction() {
  std::cout << "Streaming thread started" << std::endl;

  try {
    // Initialize sender
    if (!initialize_sender()) {
      std::cerr << "Failed to initialize sender, stopping streaming thread"
                << std::endl;
      return;
    }

    // Enable streaming flags
    encoding_enabled.store(true);
    send_enabled.store(true);

    // Initialize ZED
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = 60;
    if (zed.open(init_params) != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "Failed to open ZED camera in streaming thread\n";
      return;
    }

    // Build GStreamer pipeline
    std::string pipeline_str =
        "appsrc name=mysource is-live=true format=time "
        "caps=video/x-raw,format=BGRA,width=2560,height=720,framerate=60/1 ! "
        "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        "tee name=t "
        "t. ! queue ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true "
        "preset-level=1"
        "idrinterval=60 bitrate=4000000 ! h264parse ! appsink name=mysink "
        "emit-signals=true sync=false ";

    if (preview_enabled.load()) {
      pipeline_str += "t. ! queue ! "
                      "nvvidconv ! videoconvert ! "
                      "autovideosink sync=false ";
    }

    // Launch pipeline
    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
      std::cerr << "Failed to create pipeline in streaming thread: "
                << error->message << std::endl;
      g_clear_error(&error);
      zed.close();
      return;
    }

    // Bind appsrc/appsink
    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");

    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    sl::Mat zed_image;
    int frame_id = 0;

    std::cout << "Starting streaming loop..." << std::endl;
    while (streaming_active.load() && !stop_requested.load()) {
      if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        zed.retrieveImage(zed_image, sl::VIEW::SIDE_BY_SIDE);
        cv::Mat cv_image = slMat2cvMat(zed_image);
        if (encoding_enabled.load()) {
          GstBuffer *buffer = gst_buffer_new_allocate(
              nullptr, cv_image.total() * cv_image.elemSize(), nullptr);
          GstMapInfo map;
          gst_buffer_map(buffer, &map, GST_MAP_WRITE);
          memcpy(map.data, cv_image.data,
                 cv_image.total() * cv_image.elemSize());
          gst_buffer_unmap(buffer, &map);

          GST_BUFFER_PTS(buffer) =
              gst_util_uint64_scale(frame_id, GST_SECOND, 60);
          GST_BUFFER_DURATION(buffer) =
              gst_util_uint64_scale(1, GST_SECOND, 60);
          gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);

          frame_id++;
        }
      }
    }

    std::cout << "Streaming loop ended, cleaning up..." << std::endl;

    // Clean shutdown
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    zed.close();

  } catch (const std::exception &e) {
    std::cerr << "Streaming thread error: " << e.what() << std::endl;
  }

  std::cout << "Streaming thread finished" << std::endl;
}

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  // Check for command line arguments
  bool preview_enabled_local = false;
  bool listen_enabled = false;
  bool send_enabled_mode = false;
  std::string listen_address = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_enabled_local = true;
    } else if (arg == "--send") {
      send_enabled_mode = true;
    } else if (arg == "--listen" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
    } else if (arg == "--server" && i + 1 < argc) {
      send_to_server = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      send_to_port = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --preview      Enable video preview\n";
      std::cout << "  --listen       Listen to control commands\n";
      std::cout << "  --send         Enable sending encoded video over UDP\n";
      std::cout << "  --server IP    Server IP address (default: 127.0.0.1)\n";
      std::cout << "  --port PORT    Server port (default: 12345)\n";
      std::cout << "  --help         Show this help message\n";
      return 0;
    }
  }

  if (!listen_enabled && !send_enabled_mode) {
    std::cerr << "Error: Either --listen or --send option is required"
              << std::endl;
    std::cerr << "Use --help to see usage options" << std::endl;
    return -1;
  }

  if (send_enabled_mode && (send_to_server.empty() || send_to_port == 0)) {
    std::cerr << "Error: --send mode requires both --server and --port options"
              << std::endl;
    std::cerr << "Use --help to see usage options" << std::endl;
    return -1;
  }

  // Set global preview flag
  preview_enabled.store(preview_enabled_local);

  if (send_enabled_mode) {
    std::cout << "Starting direct video streaming to " << send_to_server << ":"
              << send_to_port << "..." << std::endl;

    // Start streaming directly
    startStreamingThread();

    // Main thread waits for termination signal
    std::cout << "Streaming started. Press Ctrl+C to stop." << std::endl;

    while (!stop_requested.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } else if (listen_enabled) {
    std::cout << "Starting listen mode on " << listen_address << "..."
              << std::endl;

    // Start listen thread
    listen_thread =
        make_unique_helper<std::thread>(listenThreadFunction, listen_address);

    // Main thread waits for termination signal
    std::cout << "Server listening. Press Ctrl+C to stop." << std::endl;

    while (!stop_requested.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Wait for listen thread to finish
    if (listen_thread && listen_thread->joinable()) {
      listen_thread->join();
      listen_thread = nullptr;
    }
  }

  std::cout << "Shutting down..." << std::endl;

  // Stop streaming thread first
  stopStreamingThread();

  std::cout << "All threads stopped. Exiting." << std::endl;
  return 0;
}